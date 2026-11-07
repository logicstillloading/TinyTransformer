"""
Deterministic N-Gram Hash Engram Table (Option B).
- Deterministic Bigram & Trigram Token Hashing:
    h2(t_{i-1}, t_i) = (t_{i-1} * 16384 + t_i) % M
    h3(t_{i-2}, t_{i-1}, t_i) = (t_{i-2} * 16384^2 + t_{i-1} * 16384 + t_i) % M
- O(1) EmbeddingBag aggregation over bigram & trigram hash slots.
- Stationary memory: consecutive tokens always hash to the exact same slot.
- Zero cluster routing drift, zero 15.4 GB/step memory gather loop, zero aux loss.
- RMSNorm on retrieved vectors before val_proj.
- Dynamic context-aware gating with active floor:
    gate = 0.10 + 0.90 * sigmoid(gate_proj(x_norm))
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast weight to x.dtype to invoke fused native C++ kernel and preserve bfloat16
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        try:
            return F.rms_norm(x, (self.dim,), w, self.eps)
        except Exception:
            variance = x.float().pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + self.eps)).to(x.dtype) * w


class NGramHashEngramTable(nn.Module):
    """
    Deterministic N-Gram Hash Engram Table.
    Uses polynomial token hashing for Bigram and Trigram n-grams into a stationary memory table.
    """
    def __init__(
        self,
        num_clusters: int = 1900,
        slots_per_cluster: int = 64,
        hidden_dim: int = 640,
        top_k: int = 4,              # Kept for signature compatibility
        chunk_size: int = 128,        # Kept for signature compatibility
        num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        if num_embeddings is not None:
            self.num_embeddings = num_embeddings
        else:
            self.num_embeddings = num_clusters * slots_per_cluster

        # Stationary memory table in FP32 master weights during training
        # Initialized with std=0.02 so variance is 4e-4 >> eps (1e-5), preventing 316x RMSNorm gradient explosion
        self.memory_table = nn.Parameter(
            torch.randn(self.num_embeddings, hidden_dim) * 0.02
        )

        # Projections & normalization
        self.norm = RMSNorm(hidden_dim)
        self.retrieved_norm = RMSNorm(hidden_dim)
        self.val_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj._is_residual = True

        # Dynamic context-aware sigmoid gating (named gating_proj to avoid substring collision with ffn.gate_proj)
        self.gating_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.zeros_(self.gating_proj.weight)
        nn.init.constant_(self.gating_proj.bias, -2.0)

        # Persistent registered zero buffer to avoid CPU scalar allocations & host-to-device kernel dispatch
        self.register_buffer("zero_loss", torch.zeros((), dtype=torch.float32), persistent=False)

    @property
    def gate_proj(self):
        """Backwards compatibility alias for gating_proj."""
        return self.gating_proj

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, T, D)
        B, T, D = x.shape
        if T == 0 or input_ids is None:
            return x, self.zero_loss

        # In case input_ids is on another device or has different length
        if input_ids.device != x.device:
            input_ids = input_ids.to(x.device)

        # Slicing optimization: for autoregressive decode (T=1) or arbitrary T,
        # we only need at most T+2 tokens from past context to compute bigrams & trigrams
        if input_ids.shape[1] > T + 2:
            input_ids = input_ids[:, -(T + 2):]

        T_ids = input_ids.shape[1]
        if T_ids >= T:
            # Full sequence context available
            pad_1 = F.pad(input_ids, (1, 0), value=0)[:, :-1]
            pad_2 = F.pad(input_ids, (2, 0), value=0)[:, :-2]
            curr_t = input_ids[:, -T:].long()
            prev_t1 = pad_1[:, -T:].long()
            prev_t2 = pad_2[:, -T:].long()
        else:
            curr_t = input_ids.long()
            prev_t1 = F.pad(curr_t, (1, 0), value=0)[:, :-1]
            prev_t2 = F.pad(curr_t, (2, 0), value=0)[:, :-2]

        # Deterministic polynomial n-gram hash modulo M
        # Uses coprime odd primes and domain separation to completely eliminate bucket collapse
        h2 = (prev_t1 * 1000003 + curr_t) % self.num_embeddings
        h3 = (prev_t2 * 1000033 + prev_t1 * 1000003 + curr_t + (self.num_embeddings // 2)) % self.num_embeddings

        # Stack into (B*T, 2) for fused EmbeddingBag
        indices = torch.stack([h2, h3], dim=-1).reshape(B * T, 2)

        # O(1) embedding bag lookup in matching compute dtype
        mem_weight = self.memory_table.to(x.dtype) if self.memory_table.dtype != x.dtype else self.memory_table
        retrieved_flat = F.embedding_bag(indices, mem_weight, mode="mean")
        retrieved = retrieved_flat.view(B, T, D)

        # Apply RMSNorm and non-linear projection on retrieved vectors
        retrieved_normed = self.retrieved_norm(retrieved)
        val_out = F.silu(self.val_proj(retrieved_normed))
        out = self.out_proj(val_out)

        # Dynamic context-aware sigmoid gating with active floor:
        # gate = 0.10 + 0.90 * sigmoid(gating_proj(x_norm))
        x_norm = self.norm(x)
        gate_raw = torch.sigmoid(self.gating_proj(x_norm))
        gate = 0.10 + 0.90 * gate_raw

        output_tensor = x + (gate.to(x.dtype) * out.to(x.dtype))
        return output_tensor, self.zero_loss


# Backwards compatibility alias
HierarchicalEngramTable = NGramHashEngramTable
