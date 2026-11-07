"""
Auxiliary 4-Layer Critic Model (~9.6M params) for adversarial BS-detection.
Discriminates between genuine data token embeddings and generator hallucinated/sampled embeddings.
Provides token-level and sequence-level discrimination scores to guide the generator via RL/adversarial reward.
"""

import math
from typing import Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engram import RMSNorm


class CriticBlock(nn.Module):
    def __init__(self, hidden_dim: int = 384, num_heads: int = 6, intermediate_dim: int = 1344):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm2 = RMSNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # Self-Attention
        norm_x = self.norm1(x)
        q = self.q_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(attn)

        # SwiGLU FFN
        norm_x2 = self.norm2(x)
        ffn = self.down_proj(F.silu(self.gate_proj(norm_x2)) * self.up_proj(norm_x2))
        x = x + ffn
        return x


class CriticModel(nn.Module):
    """
    Auxiliary 4-layer Transformer critic model (~9.6M params).
    Classifies token embeddings as genuine (1.0) vs generated/perturbed (0.0).
    """
    def __init__(
        self,
        in_dim: int = 640,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        intermediate_dim: int = 1344,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([
            CriticBlock(hidden_dim, num_heads, intermediate_dim) for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1, bias=False)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_state_dict(self, state_dict, strict: bool = True):
        """Allows loading checkpoints across different context lengths without crashing."""
        if "pos_emb" in state_dict:
            ckpt_pos = state_dict["pos_emb"]
            curr_len = self.pos_emb.shape[1]
            ckpt_len = ckpt_pos.shape[1]
            if ckpt_len != curr_len:
                new_pos = self.pos_emb.data.clone()
                copy_len = min(curr_len, ckpt_len)
                new_pos[:, :copy_len, :] = ckpt_pos[:, :copy_len, :]
                state_dict["pos_emb"] = new_pos
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, T, in_dim) token representations
        Returns:
            token_scores: (B, T) logit scores per token
            seq_score: (B,) mean logit score across the sequence
        """
        B, T, D = embeddings.shape
        if embeddings.dtype != self.in_proj.weight.dtype:
            embeddings = embeddings.to(self.in_proj.weight.dtype)
        x = self.in_proj(embeddings)
        pos = self.pos_emb[:, :T, :]
        if pos.dtype != x.dtype:
            pos = pos.to(x.dtype)
        x = x + pos

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        token_logits = self.score_head(x).squeeze(-1)  # (B, T)
        seq_logits = token_logits.mean(dim=-1)          # (B,)
        return token_logits, seq_logits


def compute_critic_loss(
    critic: CriticModel,
    real_embeds: torch.Tensor,
    fake_embeds: torch.Tensor,
    label_smoothing: bool = True,
) -> Tuple[torch.Tensor, Any, Any]:
    """
    Trains the critic to assign 0.9 to real tokens and 0.1 to fake tokens (one-sided label smoothing)
    to prevent discriminator overconfidence and saturation.
    Uses detached embeddings so generator is not updated during critic step.
    Optimized for zero device-to-host stalling on TPU / XLA.
    """
    is_xla = "xla" in str(real_embeds.device.type)
    if is_xla:
        real_token_logits, real_seq_logits = critic(real_embeds.detach())
        fake_token_logits, fake_seq_logits = critic(fake_embeds.detach())
    else:
        autocast_device = "cuda" if "cuda" in str(real_embeds.device.type) else "cpu"
        with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
            real_token_logits, real_seq_logits = critic(real_embeds.detach())
            fake_token_logits, fake_seq_logits = critic(fake_embeds.detach())

    real_target = torch.full_like(real_seq_logits, 0.9) if label_smoothing else torch.ones_like(real_seq_logits)
    fake_target = torch.full_like(fake_seq_logits, 0.1) if label_smoothing else torch.zeros_like(fake_seq_logits)

    loss_real = F.binary_cross_entropy_with_logits(real_seq_logits.float(), real_target)
    loss_fake = F.binary_cross_entropy_with_logits(fake_seq_logits.float(), fake_target)
    total_loss = 0.5 * (loss_real + loss_fake)

    if is_xla:
        real_acc = (torch.sigmoid(real_seq_logits.float()) > 0.5).float().mean()
        fake_acc = (torch.sigmoid(fake_seq_logits.float()) < 0.5).float().mean()
    else:
        real_acc = (torch.sigmoid(real_seq_logits.float()) > 0.5).float().mean().item()
        fake_acc = (torch.sigmoid(fake_seq_logits.float()) < 0.5).float().mean().item()

    return total_loss, real_acc, fake_acc


def compute_generator_adversarial_loss(
    critic: CriticModel,
    fake_embeds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generator attempts to fool the critic into believing fake embeddings are real (1.0).
    Returns adversarial loss and reward bonus.
    """
    is_xla = "xla" in str(fake_embeds.device.type)
    if is_xla:
        token_logits, seq_logits = critic(fake_embeds)
    else:
        autocast_device = "cuda" if "cuda" in str(fake_embeds.device.type) else "cpu"
        with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
            token_logits, seq_logits = critic(fake_embeds)
    # Target is 1.0 (real)
    loss_adv = F.binary_cross_entropy_with_logits(
        seq_logits.float(), torch.ones_like(seq_logits, dtype=torch.float32)
    )
    reward = torch.sigmoid(seq_logits.float()).detach()
    return loss_adv, reward
