"""
Keller Jordan's Muon Optimizer with 5th-order Newton-Schulz iteration
paired with AdamW for embeddings, 1D parameters, and Engram memory tables.
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer


def zeroth_power_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Applies 5 iterations of the Newton-Schulz method to compute the zeroth power
    (polar factor / orthogonalization) of 2D matrix or batched 3D tensor G.
    Coefficients: a = 3.4445, b = -4.7750, c = 2.0315.
    Uses fused cuBLAS baddbmm operations with float32 norm scaling and minimal allocations.
    """
    orig_ndim = G.ndim
    if orig_ndim == 2:
        G = G.unsqueeze(0)
    elif orig_ndim != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got {orig_ndim}D tensor")

    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = G.dtype

    compute_dtype = torch.bfloat16 if (G.is_cuda and torch.cuda.is_bf16_supported()) else G.dtype
    norm = torch.linalg.vector_norm(G, dim=(-2, -1), keepdim=True, dtype=torch.float32) + eps
    X = (G.float() / norm).to(compute_dtype)

    # If matrix is tall, transpose so inner dimension is smaller for matrix multiplications
    is_tall = X.size(1) > X.size(2)
    if is_tall:
        X = X.transpose(-2, -1)

    for _ in range(steps):
        A = torch.bmm(X, X.transpose(-2, -1))
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)

    if is_tall:
        X = X.transpose(-2, -1)

    X = X.to(orig_dtype)
    return X.squeeze(0) if orig_ndim == 2 else X


def _apply_update(p: torch.Tensor, ortho: torch.Tensor, lr: float, scale: float, wd: float):
    """
    Applies updates in float32 with stochastic rounding for bfloat16 parameters.
    Prevents underflow and ensures weight decay (1 - lr*wd) actually registers
    instead of rounding away in pure bf16.
    """
    delta = ortho.float().mul_(-lr * scale)
    if wd != 0.0:
        delta.add_(p.float(), alpha=-lr * wd)
    if p.dtype != torch.bfloat16:
        p.add_(delta.to(p.dtype))
        return
    x = p.float().add_(delta)
    bits = x.view(torch.int32)
    bits.add_(torch.randint_like(bits, 0, 1 << 16))  # stochastic rounding
    bits.bitwise_and_(-65536)
    p.copy_(x)


class Muon(Optimizer):
    """
    Muon (MomentUm Orthogonalized by Newton-Schulz) optimizer.
    Optimizes 2D hidden layer weight matrices by orthogonalizing momentum updates.
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            # Group parameters by (device, shape) for batched 3D Newton-Schulz
            shape_map = {}
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.ndim != 2:
                    raise ValueError(
                        f"Muon optimizer only supports 2D weight matrices. Got tensor of shape {p.shape}"
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = torch.add(g, buf, alpha=momentum)
                else:
                    update = buf

                # Keller Jordan aspect ratio scale: max(1, d_out / d_in) ** 0.5
                # Normalizes Frobenius norm for tall up-projections without over-scaling down-projections
                if "aspect_ratio_scale" not in state:
                    state["aspect_ratio_scale"] = max(1.0, p.size(0) / p.size(1)) ** 0.5
                aspect_ratio_scale = state["aspect_ratio_scale"]

                key = (p.device, p.shape)
                if key not in shape_map:
                    shape_map[key] = []
                shape_map[key].append((p, update, aspect_ratio_scale))

            # Run batched Newton-Schulz per shape group in small chunks (8) to eliminate VRAM spikes
            max_ns_batch = 8
            for (dev, shape), items in shape_map.items():
                scale = items[0][2]
                for chunk_start in range(0, len(items), max_ns_batch):
                    chunk_items = items[chunk_start : chunk_start + max_ns_batch]
                    if len(chunk_items) == 1:
                        p, update, _ = chunk_items[0]
                        ortho_update = zeroth_power_via_newtonschulz5(update, steps=ns_steps)
                        _apply_update(p, ortho_update, lr, scale, weight_decay)
                    else:
                        updates_3d = torch.stack([u for _, u, _ in chunk_items], dim=0)
                        ortho_3d = zeroth_power_via_newtonschulz5(updates_3d, steps=ns_steps)
                        for i, (p, _, _) in enumerate(chunk_items):
                            _apply_update(p, ortho_3d[i], lr, scale, weight_decay)

        return loss


def create_dual_optimizers(
    model: nn.Module,
    critic: nn.Module = None,
    lr_muon: float = 0.02,
    lr_adam: float = 3e-4,
    critic_lr: float = 1e-4,
    weight_decay: float = 0.01,
):
    """
    Separates model parameters into:
    1. Muon: All 2D internal linear weight matrices (attention, FFN gate/up/down, engram routing/output projections).
    2. AdamW:
       - Decayed: 2D embeddings (weight_decay applied).
       - Non-decayed: memory_table, centroids, 1D RMSNorm scale parameters, gate scalar, biases (weight_decay = 0.0).
    3. Critic Optimizer (if critic is provided): AdamW for the auxiliary critic model.
    """
    muon_params = []
    adam_decay_params = []
    adam_nodecay_params = []
    adam_memory_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Strictly distinguish Engram gating (gating_proj or .gate) from Transformer FFN gate_proj
        is_adam_explicit = (
            "tok_embeddings" in name
            or "lm_head" in name
            or "memory_table" in name
            or "centroids" in name
            or "gating_proj" in name
            or ("engram" in name and "gate" in name)
            or name.endswith(".gate")
            or p.ndim != 2
        )

        if is_adam_explicit:
            if "memory_table" in name or "centroids" in name:
                # Sparse stationary memory tables: non-decayed with beta2=0.999
                adam_memory_params.append(p)
            elif (
                name.endswith(".gate")
                or "gating_proj" in name
                or ("engram" in name and "gate" in name)
                or "norm" in name
                or "ngram_conv" in name
                or p.ndim < 2
            ):
                adam_nodecay_params.append(p)
            else:
                adam_decay_params.append(p)
        else:
            # 2D Linear weight matrices: Attention (q, k, v, out) and SwiGLU FFN (gate_proj, up_proj, down_proj)
            muon_params.append(p)

    optimizers = {}

    if muon_params:
        optimizers["muon"] = Muon(
            muon_params,
            lr=lr_muon,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            weight_decay=weight_decay,
        )

    all_adam_params = adam_decay_params + adam_nodecay_params + adam_memory_params
    use_fused_adam = torch.cuda.is_available() and any(p.is_cuda for p in all_adam_params)

    if all_adam_params:
        param_groups = []
        if adam_decay_params:
            param_groups.append({"params": adam_decay_params, "weight_decay": weight_decay, "betas": (0.9, 0.95)})
        if adam_nodecay_params:
            param_groups.append({"params": adam_nodecay_params, "weight_decay": 0.0, "betas": (0.9, 0.95)})
        if adam_memory_params:
            # Sparse stationary memory table: beta2=0.999 prevents second moment collapse on unvisited slots
            param_groups.append({"params": adam_memory_params, "weight_decay": 0.0, "betas": (0.9, 0.999)})

        try:
            optimizers["adam"] = torch.optim.AdamW(
                param_groups,
                lr=lr_adam,
                eps=1e-8,
                fused=use_fused_adam,
            )
        except Exception:
            optimizers["adam"] = torch.optim.AdamW(
                param_groups,
                lr=lr_adam,
                eps=1e-8,
            )

    if critic is not None:
        optimizers["critic"] = create_critic_optimizer(
            critic=critic,
            critic_lr=critic_lr,
            weight_decay=weight_decay,
        )

    return optimizers


def create_critic_optimizer(
    critic: nn.Module,
    critic_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Creates a dedicated AdamW optimizer for the Critic model with norm-nodecay separation."""
    critic_decay = []
    critic_nodecay = []
    for name, p in critic.named_parameters():
        if not p.requires_grad:
            continue
        if "norm" in name or p.ndim < 2:
            critic_nodecay.append(p)
        else:
            critic_decay.append(p)

    critic_param_groups = []
    if critic_decay:
        critic_param_groups.append({"params": critic_decay, "weight_decay": weight_decay})
    if critic_nodecay:
        critic_param_groups.append({"params": critic_nodecay, "weight_decay": 0.0})

    all_critic = critic_decay + critic_nodecay
    use_fused_critic = torch.cuda.is_available() and any(p.is_cuda for p in all_critic)
    try:
        return torch.optim.AdamW(
            critic_param_groups,
            lr=critic_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused_critic,
        )
    except Exception:
        return torch.optim.AdamW(
            critic_param_groups,
            lr=critic_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
