import os
import json
import wandb
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from torchastic import Compass

from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from src.ae import AutoEncoder
from src.flow import Flow, image_flatten, image_unflatten, cosine_optimal_transport


def load_config_from_json(filepath: str):
    with open(filepath, "r") as json_file:
        return json.load(json_file)


def main():
    torch.manual_seed(0)
    training_config = {
        "batch_size": 32,
        "lr": 5e-4,
        "num_epochs": 100,
        "eval_interval": 100,
        "preview_path": "preview_flower",
        "wandb_project": None,
        "device": "cuda:0",
        "ckpt_path": "flowers_ckpt",
        "class_dropout_ratio": 0.1,
        "model_config": {
            "input_dim": 64,
            "output_dim": 64,
            "dim": 128,
            "num_layers": 12,
            "num_heads": 8,
            "exp_fac": 4,
            "rope_seq_length": 2048,
            "class_count": 102,
        },
        "ae_config": {
            "pixel_channels": 3,
            "bottleneck_channels": 64,
            "up_layer_blocks": [[320, 20], [256, 20], [192, 20], [128, 15], [64, 10]],
            "down_layer_blocks": [[64, 10], [128, 15], [192, 20], [256, 20], [320, 20]],
            "act_fn": "sin",
        },
        "model_checkpoint": None,
        "ae_checkpoint": "ae.pth", 
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
        ae = AutoEncoder(**training_config["ae_config"])
        ae.load_state_dict(torch.load(training_config["ae_checkpoint"], weights_only=True))
        ae.to(training_config["device"])
        ae.to(torch.bfloat16)
        ae.eval()

    optim = Compass(model.parameters(), lr=training_config["lr"], amp_fac=5)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.00001, end_factor=1.0, total_iters=100
    )

    if training_config["wandb_project"]:
        wandb.init(
            project=training_config["wandb_project"],
            name=training_config["preview_path"],
        )

    # epoch loop
    for epoch in range(training_config["num_epochs"]):
        # roll epoch
        torch.manual_seed(epoch)
        progress_bar = tqdm(total=len(loader), desc="Processing", smoothing=0.1)

        # training loop
        for batch_idx, (real, label) in enumerate(loader):
            DEVICE = training_config["device"]
            real = real.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                latent = ae.encode(real * 2 - 1) 
                
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
                f"Epoch [{epoch}/{training_config['num_epochs']}] Step [{batch_idx}/{len(loader)}]  Loss: {loss:.4f}"
            )

            if training_config["wandb_project"]:
                wandb.log(
                    {
                        "Loss": loss,
                        "Epoch": epoch,
                    }
                )
            if batch_idx % training_config["eval_interval"] == 0:

                with torch.no_grad():
                    z = torch.randn_like(real)
                    with torch.autocast("cuda", torch.bfloat16):
                        fake_latent, _ = model.euler(z, label, num_steps=8)
                        fake_latent_cfg, _ = model.euler_cfg(z, label, 3, num_steps=8)

                    fake_latent = torch.cat([fake_latent, fake_latent_cfg, real], dim=0)
                    fake_latent = image_unflatten(fake_latent, image_shape, 1)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                        fake_images = ae.decode(fake_latent) # rescale to (-1, 1)


                    if not os.path.exists(training_config["preview_path"]):
                        os.makedirs(training_config["preview_path"])

                    image_grid = make_grid((fake_images + 1) / 2, nrow=4, padding=2)
                    save_image(image_grid.clamp(0, 1), f"{training_config['preview_path']}/epoch_{epoch}_{batch_idx}.png")


            progress_bar.update(1)

        if not os.path.exists(training_config["ckpt_path"]):
            os.makedirs(training_config["ckpt_path"])
        torch.save(model.state_dict(), f"{training_config['ckpt_path']}/{epoch}.pth")


if __name__ == "__main__":
    main()
