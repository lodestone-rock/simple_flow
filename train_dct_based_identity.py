import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import wandb
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from torchastic import AdamW

from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from dct_autoencoder import DCTAutoencoder
from src.flow_identity import Flow, image_flatten, image_unflatten, cosine_optimal_transport
from diffusers import AutoencoderDC


def load_config_from_json(filepath: str):
    with open(filepath, "r") as json_file:
        return json.load(json_file)




def center_crop_to_divisible(image, block_size):
    """
    Center crops an image tensor to dimensions that are divisible by block_size.
    
    Args:
        image: A tensor of shape [C, H, W]
        block_size: The block size to make dimensions divisible by
        
    Returns:
        A center-cropped tensor with dimensions divisible by block_size
    """
    _, h, w = image.shape
    
    # Calculate new dimensions divisible by block_size
    new_h = h - (h % block_size)
    new_w = w - (w % block_size)
    
    # Calculate crop margins
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    
    # Perform center crop
    cropped_image = image[:, top:top + new_h, left:left + new_w]
    
    return cropped_image


def prune_dct(dct, dct_size, keep_y, keep_cb, keep_cr):
    dct = dct.clone()
    dct_size = dct_size ** 2

    # prune high freq and quantize the mid freq (probably quantization is unnecessary here)
    return torch.cat(
        [ 
            dct[:, 0:1], # base freq for Y 
            dct[:, 1:dct_size * 0 + keep_y].to(torch.float8_e4m3fn).to(torch.float32), # quantize high freq
            dct[:, dct_size * 1: dct_size * 1 + 1], # base freq for Cb
            dct[:, 1 + dct_size * 1:dct_size * 1 + keep_cb].to(torch.float8_e4m3fn).to(torch.float32), # quantize high freq
            dct[:, dct_size * 2: dct_size * 2 + 1], # base freq for Cr
            dct[:, 1 + dct_size * 2:dct_size * 2 + keep_cr].to(torch.float8_e4m3fn).to(torch.float32), # quantize high freq
        ],
        dim=1
    )


def pad_channels(x, pad_left=0, pad_right=0, value=0):
    # x is shape (B, C, H, W)
    x = x.permute(0, 2, 3, 1)  # to (B, H, W, C)
    x = F.pad(x, (pad_left, pad_right), value=value)  # pad channels
    x = x.permute(0, 3, 1, 2)  # back to (B, C, H, W)
    return x


def pad_dct(dct, dct_size, keep_y, keep_cb, keep_cr):
    dct = dct.clone()
    dct_size = dct_size ** 2
    return torch.cat(
        [ 
            pad_channels(dct[:, :keep_y], 0, dct_size - keep_y),
            pad_channels(dct[:, keep_y: keep_y + keep_cb], 0, dct_size - keep_cb),
            pad_channels(dct[:, keep_y + keep_cb: keep_y + keep_cb + keep_cr], 0, dct_size - keep_cr),
        ],
        dim=1
    )


def main():
    torch.manual_seed(0)
    training_config = {
        "batch_size": 32,
        "lr": 1e-4,
        "num_epochs": 100,
        "eval_interval": 250,
        "preview_path": "dct_flower_dct_identity",
        "wandb_project": "dct_flow",
        "device": "cuda:0",
        "ckpt_path": "dct_flower_dct_identity",
        "class_dropout_ratio": 0.1,
        "model_config": {
            "input_dim": 32,
            "output_dim": 32,
            "proj_dim_repeat": 512//32,
            "dim": 512,
            "num_layers": 12,
            "num_heads": 8,
            "exp_fac": 4,
            "rope_seq_length": 2048,
            "class_count": 102,
        },
        "ae_config": {
            "y_channels": 16,
            "cb_channels": 8,
            "cr_channels": 8,
            "dct_size": 32,
        },
        "model_checkpoint": None,
    }

    # Data Loader
    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(512),
            transforms.CenterCrop(512),
        ]
    )

    # Load dataset (Train split)
    dataset = datasets.Flowers102(
        root="dataset/",
        split="train",  # Use "train", "val", or "test"
        transform=transform,
        download=True,
    )
    loader = DataLoader(dataset, batch_size=training_config["batch_size"], shuffle=True)
    with torch.no_grad():
        # flow backbone
        model = Flow(**training_config["model_config"])
        if training_config["model_checkpoint"]:
            model.load_state_dict(torch.load(training_config["model_checkpoint"], weights_only=True))
        model.to(training_config["device"])
        model.to(torch.bfloat16)
        # model.transformer.set_use_compiled()
        model.train()

        # ae
        ae = DCTAutoencoder(
            block_size=training_config["ae_config"]["dct_size"],
            luminance_compression_ratio=1,
            chrominance_compression_ratio=1,
        )
        ae.to(training_config["device"])

    optim = AdamW(model.parameters(), lr=training_config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.00001, end_factor=1.0, total_iters=5
    )

    if training_config["wandb_project"]:
        wandb.init(
            project=training_config["wandb_project"],
            name=training_config["preview_path"],
        )

    # epoch loop
    epoch = 0
    counter = 0
    # for epoch in range(training_config["num_epochs"]):
    while True:
        epoch += 1
        # roll epoch
        torch.manual_seed(epoch)
        progress_bar = tqdm(total=len(loader), desc="Processing", smoothing=0.1)

        # training loop
        for batch_idx, (real, label) in enumerate(loader):
            DEVICE = training_config["device"]
            real = real.to(DEVICE)
            with torch.no_grad():
                latent = ae.encode(real) 
                latent = prune_dct(
                    latent, 
                    training_config["ae_config"]["dct_size"], 
                    training_config["ae_config"]["y_channels"], 
                    training_config["ae_config"]["cb_channels"], 
                    training_config["ae_config"]["cr_channels"]
                ).to(torch.bfloat16)

            # flatten the image
            real, image_shape = image_flatten(latent, 1)
            real = real.requires_grad_(True)
            label = label.to(DEVICE)
            B = real.shape[0]

            # noise pairings
            x0 = torch.randn_like(real)
            transport_cost, indices = cosine_optimal_transport(
                real.reshape(B, -1), x0.reshape(B, -1)
            )
            x0 = x0[indices[1].view(-1)]

            with torch.autocast("cuda", torch.bfloat16):
                # compute loss
                loss = model.loss_rectified_flow(
                    batch=real,
                    class_label=label,
                    x0=x0,
                    class_dropout_ratio=training_config["class_dropout_ratio"],
                )
            loss.backward()
            optim.step()
            lr_scheduler.step()
            optim.zero_grad()

            progress_bar.set_description(
                f"Epoch [{epoch}] Step [{batch_idx}/{len(loader)}]  Loss: {loss:.4f}"
            )

            if training_config["wandb_project"]:
                wandb.log(
                    {
                        "Loss": loss,
                        "Epoch": epoch,
                    }
                )
            if counter % training_config["eval_interval"] == 0:

                with torch.no_grad():
                    z = torch.randn_like(real[:4])
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                        fake_latent, _ = model.euler(z, label[:4], num_steps=8)
                        fake_latent_cfg, _ = model.euler_cfg(z, label[:4], 3, num_steps=8)

                    fake_latent = torch.cat([fake_latent, fake_latent_cfg, real[:4]], dim=0)
                    fake_latent = image_unflatten(fake_latent, image_shape, 1)

                    fake_latent = pad_dct(
                        fake_latent, 
                        training_config["ae_config"]["dct_size"], 
                        training_config["ae_config"]["y_channels"], 
                        training_config["ae_config"]["cb_channels"], 
                        training_config["ae_config"]["cr_channels"]
                    )
                    fake_images = ae.decode(fake_latent.to(torch.float32))

                    if not os.path.exists(training_config["preview_path"]):
                        os.makedirs(training_config["preview_path"])

                    image_grid = make_grid(fake_images, nrow=4, padding=2)
                    save_image(image_grid.clamp(0, 1), f"{training_config['preview_path']}/epoch_{epoch}_{batch_idx}.jpg")


            progress_bar.update(1)
            counter += 1

        if counter % training_config["eval_interval"] == 0:
            if not os.path.exists(training_config["ckpt_path"]):
                os.makedirs(training_config["ckpt_path"])
            torch.save(model.state_dict(), f"{training_config['ckpt_path']}/{epoch}.pth")


if __name__ == "__main__":
    main()
