import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .backbone import TransformerNetwork, GLU, SoftClamp
import math
from tqdm import tqdm

# cuda impl of hungarian method
from torch_linear_assignment import batch_linear_assignment
from torch_linear_assignment import assignment_to_indices


def image_flatten(latents, shuffle_size=2):
    # nchw to nhwc then pixel shuffle of arbitrary size then flatten
    # n c h w -> n h w c
    # n (h dh) (w dw) c -> n h w (c dh dw)
    # n h w c -> n (h w) c
    return (
        rearrange(
            latents,
            "n c (h dh) (w dw) -> n (h w) (c dh dw)",
            dh=shuffle_size,
            dw=shuffle_size,
        ),
        latents.shape,
    )


def image_unflatten(latents, shape, shuffle_size=2):
    # reverse of the flatten operator above
    n, c, h, w = shape
    return rearrange(
        latents,
        "n (h w) (c dh dw) -> n c (h dh) (w dw)",
        dh=shuffle_size,
        dw=shuffle_size,
        c=c,
        h=h // shuffle_size,
        w=w // shuffle_size,
    )


def sample_from_distribution(x, probabilities, n):
    indices = torch.multinomial(probabilities, n, replacement=True)
    return x[indices]


def create_distribution(num_points, device=None):
    # Probability range on x axis
    x = torch.linspace(0, 1, num_points, device=device)

    # Custom probability density function
    probabilities = -7.7 * ((x - 0.5) ** 2) + 2

    # Normalize to sum to 1
    probabilities /= probabilities.sum()

    return x, probabilities


def repeat_along_dim(tensor, repeats, dim):
    # Move the desired dimension to the front
    permute_order = list(range(tensor.dim()))
    permute_order[dim], permute_order[0] = permute_order[0], permute_order[dim]
    tensor = tensor.permute(permute_order)

    # Unsqueeze to add a new dimension for repetition
    tensor = tensor.unsqueeze(1)

    # Repeat along the new dimension
    repeated_tensor = tensor.repeat(1, repeats, *([1] * (tensor.dim() - 2)))

    # Collapse the repeated dimension
    repeated_tensor = repeated_tensor.view(-1, *repeated_tensor.shape[2:])

    # Move the dimension back to its original order
    permute_order[dim], permute_order[0] = permute_order[0], permute_order[dim]
    repeated_tensor = repeated_tensor.permute(permute_order)

    return repeated_tensor


def cosine_optimal_transport(X, Y):
    """
    Compute optimal transport between two sets of vectors using cosine distance.

    Parameters:
    X: torch.Tensor of shape (n, d)
    Y: torch.Tensor of shape (m, d)

    Returns:
    P: optimal transport plan matrix of shape (n, m)
    """
    # Normalize input vectors
    X_norm = X / torch.norm(X, dim=1, keepdim=True)
    Y_norm = Y / torch.norm(Y, dim=1, keepdim=True)

    # Compute cost matrix using matrix multiplication (cosine similarity)
    C = -torch.mm(X_norm, Y_norm.t())  # negative because we want to minimize distance

    assignment = batch_linear_assignment(C.unsqueeze(dim=0))
    matching_pairs = assignment_to_indices(assignment)

    return C, matching_pairs


class Flow(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        proj_dim_repeat,
        dim,
        num_layers,
        num_heads=8,
        exp_fac=4,
        rope_seq_length=784,
        class_count=10,
    ):
        super(Flow, self).__init__()

        self.proj_dim_repeat = proj_dim_repeat
        # disable input projection init on TransformerNetwork and init it here instead
        # TransformerNetwork should act as its own discretize ODE by itself (resnet nature of the model)
        # so technically we don't need to have really long ODE stepping!
        self.transformer = TransformerNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            exp_fac=exp_fac,
            rope_seq_length=rope_seq_length,
            final_head=False,
            # set input projection to false because we want to
            # concatenate time token after image projection
            input_proj=False,
        )
        self.dim = dim
        self.timestep_proj = GLU(
            dim, exp_fac=2
        )  # reusing GLU as feed forward replacement
        self.timestep_vector = nn.Linear(1, dim)
        nn.init.zeros_(self.timestep_vector.weight)
        nn.init.zeros_(self.timestep_vector.bias)
        # extra embed for wildcard
        self.class_embed = nn.Linear(1, dim)
        self.class_norm = SoftClamp(dim=dim)
        self.class_count = class_count

        nn.init.zeros_(self.class_embed.weight)
        nn.init.zeros_(self.class_embed.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, class_embed, attention_mask=None):
        b, l, d = x.shape
        B = x.shape[0]
        # timesteps [B, 1]
        # use simple projection here, nothing fancy
        # project image patch to model dim
        x = x.repeat_interleave(self.proj_dim_repeat, dim=-1)
        # concatenate time token and use it as a guidance vector
        class_vec = self.class_embed(class_embed.to(self.class_embed.weight)[:,None,None,])
        class_vec = self.class_norm(class_vec)
        # class_vec = self.class_embed(class_embed.to(self.class_embed.weight))[:, None, :]
        # x = torch.cat((time_token, class_vec, x), dim=1)
        x = torch.cat((class_vec, x), dim=1)
        # Forward through transformer network
        x = self.transformer(x, attention_mask)
        b, l, d2 = x.shape
        x = x.view(b, l, d2 // self.proj_dim_repeat, self.proj_dim_repeat).mean(dim=-1)
        # remove the guidance vector
        x_out = x[:, 1:, ...]

        return x_out

    def loss_rectified_flow(self, batch, class_label, x0, class_dropout_ratio=0.1):

        class_label_clone = class_label.clone()
        B = batch.shape[0]
        class_mask = torch.rand(B) < class_dropout_ratio
        class_label_clone[class_mask] = self.class_count  # wildcard class
        # noise
        # x0 = torch.randn_like(batch) * gaussian_prior_scale
        # using rounding to make the problem discrete-ish

        num_points = 1000  # Number of points in the range
        x, probabilities = create_distribution(num_points, device=self.device)
        t = sample_from_distribution(x, probabilities, B)

        # t = F.sigmoid(torch.randn((B,), device=self.device))
        # Compute noised data points
        xt = x0 * (1 - t[:, None, None]) + batch * t[:, None, None]
        # Forward pass and compute noise to image vector
        v = self.forward(xt, class_label_clone)
        # mse loss

        return torch.mean(torch.mean((v - (batch - x0)) ** 2, dim=(1, 2)))

    def euler(self, x, cond, num_steps=10, skip_last_n=0, return_intermediates=False):
        t = torch.linspace(0, 1, num_steps).to(self.device)
        if return_intermediates:
            trajectories = [x.cpu()]
        else:
            trajectories = None

        for i in tqdm(range(1, num_steps)):
            if ((num_steps - i) - skip_last_n) == 0:
                break
            with torch.no_grad():
                v = self.forward(x, cond)
                x = x + v * (t[i] - t[i - 1])

            if return_intermediates:
                trajectories.append(x.cpu())

        return x, trajectories

    def euler_cfg(
        self,
        x,
        pos_cond,
        cfg_scale=1,
        num_steps=10,
        skip_last_n=0,
        return_intermediates=False,
    ):
        t = torch.linspace(0, 1, num_steps).to(self.device)
        if return_intermediates:
            trajectories = [x.cpu()]
        else:
            trajectories = None

        neg_cond = torch.tensor([self.class_count] * pos_cond.shape[0]).to(pos_cond)
        for i in tqdm(range(1, num_steps)):
            if ((num_steps - i) - skip_last_n) == 0:
                break
            with torch.no_grad():
                v = self.forward(x, pos_cond)
                v_neg = self.forward(x, neg_cond)
                v = v + cfg_scale * (v - v_neg)
                x = x + v * (t[i] - t[i - 1])

            if return_intermediates:
                trajectories.append(x.cpu())

        return x, trajectories
