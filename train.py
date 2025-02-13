import os
import wandb
from tqdm import tqdm


import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from torchvision.utils import save_image
from torchvision import datasets, transforms

from src.flow import Flow, image_flatten, image_unflatten, cosine_optimal_transport
from torchastic import Compass

torch.manual_seed(0)
training_config = {
    "batch_size": 64,
    "lr": 5e-4,
    "num_epochs": 100,
    "eval_interval": 100,
    "preview_path": "preview",
    "wandb_project": "flow",
    "device": "cuda:0",
    "ckpt_path": "flow_ckpt",
    "implicit_factor": 10,
    "class_dropout_ratio": 0.1,
}

# Data Loader
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

dataset = datasets.CIFAR10(
    root="mnist/", train=True, transform=transform, download=True
)
loader = DataLoader(dataset, batch_size=training_config["batch_size"], shuffle=True)

model = Flow(
    input_dim=12,
    output_dim=12,
    dim=128,
    num_layers=10,
    num_heads=8,
    exp_fac=4,
    rope_seq_length=784,
)
model.to(training_config["device"])
model.to(torch.bfloat16)
model.transformer.set_use_compiled()

optim = Compass(model.parameters(), lr=training_config["lr"], amp_fac=5)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    optim, start_factor=0.00001, end_factor=1.0, total_iters=100
)


wandb.init(
    project=training_config["wandb_project"], name=training_config["preview_path"]
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
        real, image_shape = image_flatten(real)
        real = real.requires_grad_(True)
        label = label.to(DEVICE)
        B = real.shape[0]

        x0 = torch.randn_like(real)
        # if we're echoing then we better use optimal transport pairing
        transport_cost, indices = cosine_optimal_transport(
            real.reshape(B, -1), x0.reshape(B, -1)
        )
        x0 = x0[indices[1].view(-1)]

        with torch.autocast("cuda", torch.bfloat16):
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
                    fake_images, _ = model.euler(z, label, num_steps=30)
                    fake_images_cfg, _ = model.euler_cfg(z, label, 3, num_steps=30)

                fake_images = torch.cat([fake_images, fake_images_cfg, real], dim=0)
                fake_images = image_unflatten(fake_images, image_shape)
                if not os.path.exists(training_config["preview_path"]):
                    os.makedirs(training_config["preview_path"])
                save_image(
                    torchvision.utils.make_grid(fake_images.clip(-1, 1)),
                    f"{training_config['preview_path']}/epoch_{epoch}_{batch_idx}.png",
                )

        progress_bar.update(1)

    if not os.path.exists(training_config["ckpt_path"]):
        os.makedirs(training_config["ckpt_path"])
    torch.save(model.state_dict(), f"{training_config['ckpt_path']}/{epoch}.pth")
