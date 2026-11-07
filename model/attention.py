"""
Multi-Head Attention (MHA) with QK-RMSNorm and Rotary Position Embeddings (RoPE).
Standard full-rank attention across uniform attention heads:
- Pre-RMSNorm on input representations.
- Per-head QK-RMSNorm to eliminate score explosion without artificial logit clipping.
- Rotary Position Embeddings (RoPE) and PyTorch's native Flash/SDPA attention.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engram import RMSNorm


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # inv_freq shape: (dim // 2,)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (max_seq_len, dim // 2)
        # Embs shape: (max_seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x_or_dtype, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = x_or_dtype if isinstance(x_or_dtype, torch.dtype) else x_or_dtype.dtype
        if offset + seq_len > self.cos_cached.size(0):
            self._build_cache(max(offset + seq_len, self.cos_cached.size(0) * 2))
        cos = self.cos_cached[offset : offset + seq_len].to(dtype)
        sin = self.sin_cached[offset : offset + seq_len].to(dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, num_heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim) -> unsqueeze to (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention (MHA) with Rotary Position Embeddings (RoPE):
    - 8 uniform attention heads of dimension 64 (512 total hidden dimension).
    - Full rank representation for Queries, Keys, and Values (no latent compression bottleneck).
    - Uniform 64-dim RoPE positional encoding across query and key heads.
    - Native PyTorch SDPA (FlashAttention kernel) execution.
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
        latent_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        rope: Optional[RotaryEmbedding] = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        assert num_heads * head_dim == hidden_dim, f"{num_heads}x{head_dim} != {hidden_dim}"

        # Standard full-rank query, key, value, and output projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # QK-Norm: normalizes queries and keys per-head before dot product to prevent score explosion
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # Uniform 64-dim Rotary Embedding (shared instance if provided)
        if rope is not None:
            self.rope = rope
        else:
            self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, D = x.shape

        # 1. Project Q, K, V with QK-Norm: (B, T, D) -> (B, num_heads, T, head_dim)
        q = self.q_norm(self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2))
        k = self.k_norm(self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2))
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Apply Uniform 64-dim RoPE to NEW keys/queries before cache concatenation
        past_len = 0
        prev_k = prev_v = None
        if kv_cache is not None:
            if isinstance(kv_cache[0], (tuple, list)):
                prev_k, prev_v = kv_cache[0]
            else:
                prev_k, prev_v = kv_cache
            past_len = prev_k.shape[2]

        if rope_cos is None or rope_sin is None:
            cos, sin = self.rope(q, T, offset=past_len)
        else:
            cos, sin = rope_cos, rope_sin

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)  # rotate NEW keys only

        # 3. Concatenate past KV cache with newly rotated K and V
        if prev_k is not None:
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        # 4. Scaled Dot-Product Attention (FlashAttention kernel)
        is_causal = (past_len == 0 and T > 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # 5. Reshape and project out: (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out, new_kv_cache


# Backwards compatibility aliases
MultiHeadLatentAttention = MultiHeadAttention
HybridDualResolutionAttention = MultiHeadAttention
