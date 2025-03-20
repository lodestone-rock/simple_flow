import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math


# kernel (if you want to optimize the code, optimize all of these kernel!)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def soft_clamp(x, scale, alpha, shift):
    return scale * F.tanh(x * alpha) + shift


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (max_seq_len ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x, seq_dim=1):
        return (
            self.cos_cached[:, :, : x.shape[seq_dim], :],
            self.sin_cached[:, :, : x.shape[seq_dim], :],
        )


class SoftClamp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.use_compiled = False

    def forward(self, x):
        if self.use_compiled:
            return torch.compile(soft_clamp)(x, self.scale, self.alpha, self.shift)
        else:
            return soft_clamp(x, self.scale, self.alpha, self.shift)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, max_seq_len=2048, use_rope=True):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # enabling bias here so the model has a freedom to shift the activation
        self.wo = nn.Linear(dim, dim, bias=True)
        self.layer_norm = SoftClamp(dim)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        self.q_norm = SoftClamp(dim)
        self.k_norm = SoftClamp(dim)

        self.add_module("layer_norm", self.layer_norm)

        nn.init.zeros_(self.wo.weight)
        self.use_compiled = False
        self.use_rope = use_rope

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.layer_norm(x)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(
            1, 2
        )

        cos, sin = self.rope(x, seq_dim=1)
        if self.use_rope:
            if self.use_compiled:
                q, k = torch.compile(apply_rotary_pos_emb)(q, k, cos, sin)
            else:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if attention_mask is not None:
            # Ensure the mask is broadcastable to the attention shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            if attention_mask.dtype is not torch.bool:
                attention_mask = (1.0 - attention_mask) * torch.finfo(q.dtype).min

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask
        )
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        out = self.wo(out)

        return out + residual


class GLU(nn.Module):
    def __init__(self, dim, exp_fac=4):
        super(GLU, self).__init__()
        self.wi_0 = nn.Linear(dim, dim * exp_fac, bias=False)
        self.wi_1 = nn.Linear(dim, dim * exp_fac, bias=False)
        # enabling bias here so the model has a freedom to shift the activation
        self.wo = nn.Linear(dim * exp_fac, dim, bias=True)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        nn.init.zeros_(self.wo.weight)
        self.use_compiled = False

    @property
    def device(self):
        return next(self.parameters()).device

    def _fwd_glu(self, x, residual):
        return self.wo(F.silu(self.wi_0(x)) * self.wi_1(x)) + residual

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        if self.use_compiled:
            return torch.compile(self._fwd_glu)(x, residual)
        else:
            return self._fwd_glu(x, residual)


class TransformerNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dim,
        num_layers,
        num_heads=8,
        exp_fac=4,
        rope_seq_length=2048,
        use_rope=True,
        final_head=True,
        input_proj=True,
    ):
        super(TransformerNetwork, self).__init__()
        if input_proj:
            self.input_layer = nn.Linear(input_dim, dim)
        else:
            self.input_layer = nn.Identity()
            input_dim = dim
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": AttentionBlock(
                            dim, num_heads, rope_seq_length, use_rope
                        ),
                        "glu": GLU(dim, exp_fac),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = SoftClamp(dim)
        if final_head:
            self.output_layer = nn.Linear(dim, output_dim)
        else:
            self.output_layer = nn.Identity()

    def set_use_compiled(self):
        for name, module in self.named_modules():
            # Check if the module has the 'use_compiled' attribute
            if hasattr(module, "use_compiled"):
                print(f"Setting 'use_compiled' to True in module: {name}")
                setattr(module, "use_compiled", True)

    def forward(self, x, attention_mask=None, act_ckpt=False):
        # just use checkpoint, your GPU is fast enough to recompute the entire thing
        if act_ckpt:
            x = checkpoint(self.input_layer, x)
            for block in self.blocks:
                if type(block) == nn.Identity:
                    continue
                # res = x
                x = checkpoint(
                    lambda x, mask: block["attn"](x, mask), x, attention_mask
                )
                x = checkpoint(block["glu"], x)
                # x = res + x
            x = checkpoint(self.out_norm, x)
            x = checkpoint(self.output_layer, x)

        else:
            x = self.input_layer(x)
            for block in self.blocks:
                if type(block) == nn.Identity:
                    continue
                # res = x
                x = block["attn"](x, attention_mask)
                x = block["glu"](x)
                # x = res + x
            x = self.out_norm(x)
            x = self.output_layer(x)
        return x
