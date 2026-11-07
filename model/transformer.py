"""
Complete TinyTransformer Architecture (~238M parameters in BF16).
Dense Backbone:
- Hidden dim 640, 24 layers, 10 attention heads x 64 head_dim
- SwiGLU FFN with 3.0 multiplier (intermediate dim 1920)
- Full-rank Multi-Head Attention with shared Rotary Position Embeddings (RoPE)
- Pre-RMSNorm and tied 16,384 token embeddings (~138M params)
Engram Memory Tables (~100M params):
- 80M global table (post-layer 16) + two 10M local tables (at Layer 5 and Layer 11)
- O(1) polynomial coprime hashing for bigram/trigram context with dynamic context-aware gating
- Reasoning-After-Retrieval: Layers 17..23 (7 layers) perform deep reasoning over retrieved facts
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .attention import MultiHeadAttention, MultiHeadLatentAttention, HybridDualResolutionAttention, RotaryEmbedding
from .engram import NGramHashEngramTable, HierarchicalEngramTable, RMSNorm


@dataclass
class TransformerConfig:
    vocab_size: int = 16384
    hidden_dim: int = 640
    num_layers: int = 24
    num_heads: int = 10
    head_dim: int = 64
    ffn_mult: float = 3.0
    intermediate_dim: int = 1920  # int(640 * 3.0)
    max_seq_len: int = 1024
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    # Engram configuration: Uniform spacing for 24 layers: Local 1 at Layer 5, Local 2 at Layer 11, Global at Layer 16
    engram_layers: Tuple[int, ...] = (5, 11)  # Layer indices 5 and 11 for local engrams
    local_engram_clusters: int = 220
    global_engram_clusters: int = 1900
    slots_per_cluster: int = 64
    engram_top_k: int = 4
    engram_chunk_size: int = 128

    def __post_init__(self):
        if self.head_dim == 0 or self.num_heads * self.head_dim != self.hidden_dim:
            self.head_dim = self.hidden_dim // self.num_heads
        if self.intermediate_dim == 0 or (self.intermediate_dim == 1920 and self.hidden_dim != 640):
            self.intermediate_dim = int(self.hidden_dim * self.ffn_mult)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network with intermediate_dim = int(hidden_dim * ffn_mult).
    SwiGLU(x) = (SiLU(gate(x)) * up(x)) @ down.
    """
    def __init__(self, hidden_dim: int = 512, intermediate_dim: int = 1792):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, config: TransformerConfig, rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.pre_attn_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.attn = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            rope=rope,
        )
        self.pre_ffn_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
        )

        # Mark residual output projections for scaled initialization
        self.attn.out_proj._is_residual = True
        self.ffn.down_proj._is_residual = True

        # 10M Engram Table at designated layers (Layer indices 5 and 11)
        self.has_engram = layer_idx in config.engram_layers
        if self.has_engram:
            self.engram = HierarchicalEngramTable(
                num_clusters=config.local_engram_clusters,
                slots_per_cluster=config.slots_per_cluster,
                hidden_dim=config.hidden_dim,
                top_k=config.engram_top_k,
                chunk_size=config.engram_chunk_size,
            )
            self.engram.out_proj._is_residual = True
        else:
            self.engram = None

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Any]:
        # Pre-norm Self-Attention
        attn_out, new_cache = self.attn(
            self.pre_attn_norm(x),
            kv_cache=kv_cache,
            use_cache=use_cache,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = x + attn_out

        # Pre-norm SwiGLU FFN
        x = x + self.ffn(self.pre_ffn_norm(x))

        # Local Engram Table (if present)
        if self.engram is not None:
            x, aux_loss = self.engram(x, input_ids=input_ids)
        else:
            aux_loss = 0.0

        return x, new_cache, aux_loss

    def forward_train(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        attn_out, _ = self.attn(
            self.pre_attn_norm(x),
            kv_cache=None,
            use_cache=False,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = x + attn_out
        x = x + self.ffn(self.pre_ffn_norm(x))
        if self.engram is not None:
            x, aux_loss = self.engram(x, input_ids=input_ids)
        else:
            aux_loss = 0.0
        return x, aux_loss


class TinyTransformer(nn.Module):
    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()
        self.config = config

        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Shared Rotary Embedding computed once per forward pass for all layers (eliminates 384 redundant casts/launches per step)
        self.rope = RotaryEmbedding(dim=config.head_dim, max_seq_len=config.max_seq_len)

        # 24 Transformer layers (with 10M Engram tables at layer 5 and layer 11)
        self.layers = nn.ModuleList([
            TransformerBlock(i, config, rope=self.rope) for i in range(config.num_layers)
        ])

        # 80M Global Engram Table
        self.global_engram = HierarchicalEngramTable(
            num_clusters=config.global_engram_clusters,
            slots_per_cluster=config.slots_per_cluster,
            hidden_dim=config.hidden_dim,
            top_k=config.engram_top_k,
            chunk_size=config.engram_chunk_size,
        )
        self.global_engram.out_proj._is_residual = True

        # Final RMSNorm
        self.final_norm = RMSNorm(config.hidden_dim, eps=config.norm_eps)

        # Output LM head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Persistent registered zero buffer to avoid CPU scalar allocations & host-to-device kernel dispatch
        self.register_buffer("zero_loss", torch.zeros((), dtype=torch.float32), persistent=False)

        # Mark residual output projections so ONLY they receive 1 / sqrt(2 * num_layers) scaling
        for layer in self.layers:
            layer.attn.out_proj._is_residual = True
            layer.ffn.down_proj._is_residual = True
            if layer.has_engram:
                layer.engram.out_proj._is_residual = True
        self.global_engram.out_proj._is_residual = True

        # Gradient checkpointing flag for VRAM optimization during training
        self.gradient_checkpointing = True

        # Weight initialization
        self.apply(self._init_weights)

        # Dynamic context-aware gating initialization (bias=-2.0 gives clean ~0.20 initial floor)
        for layer in self.layers:
            if layer.has_engram:
                nn.init.constant_(layer.engram.gating_proj.bias, -2.0)
                nn.init.zeros_(layer.engram.gating_proj.weight)
        nn.init.constant_(self.global_engram.gating_proj.bias, -2.0)
        nn.init.zeros_(self.global_engram.gating_proj.weight)

        # Re-tie lm_head with tok_embeddings AFTER _init_weights to preserve embedding std = 0.02
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            if getattr(module, "_is_residual", False):
                std = 0.02 / math.sqrt(2 * self.config.num_layers)
            else:
                std = 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """Calculates precise parameter counts and memory requirements."""
        total_params = sum(p.numel() for p in self.parameters())
        # Tied parameters count once in unique memory
        unique_params = sum(p.numel() for p in set(self.parameters()))

        engram_params = self.global_engram.num_parameters
        for layer in self.layers:
            if layer.has_engram:
                engram_params += layer.engram.num_parameters

        backbone_params = unique_params - engram_params
        embedding_params = self.tok_embeddings.weight.numel()

        bf16_mb = (unique_params * 2) / (1024 * 1024)

        return {
            "total_params": total_params,
            "unique_params": unique_params,
            "backbone_params": backbone_params,
            "engram_params": engram_params,
            "embedding_params": embedding_params,
            "bf16_weight_size_mb": bf16_mb,
        }

    def get_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Helper to get ground-truth token embeddings for critic discrimination."""
        return self.tok_embeddings(token_ids)

    def sample_embeddings_gumbel(self, logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        Differentiable Gumbel-Softmax straight-through sampling of token embeddings
        from generator logits, allowing gradients to propagate from Critic to Generator.
        Embeddings are detached so adversarial loss trains generator logits rather than mutating embedding weights.
        """
        probs = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        embed_weight = self.tok_embeddings.weight.detach()
        if probs.dtype != embed_weight.dtype:
            embed_weight = embed_weight.to(probs.dtype)
        sampled_embeds = torch.matmul(probs, embed_weight)
        return sampled_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        return_hidden: bool = False,
        all_input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) token indices
            kv_caches: optional list of cached (k, v) tuples for each layer
            use_cache: whether to return new KV caches
            return_hidden: whether to return pre-head hidden states
            all_input_ids: optional (B, T_total) token sequence including past context
        Returns:
            logits: (B, T, vocab_size)
            new_caches: updated KV caches if use_cache is True
            hidden_states: (B, T, hidden_dim)
            total_aux_loss: auxiliary loss across engram tables
        """
        B, T = input_ids.shape
        x = self.tok_embeddings(input_ids)

        # Ensure residual stream matches layer compute dtype (bfloat16) to avoid continuous dynamic casts
        target_dtype = self.layers[0].attn.q_proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        full_ids = all_input_ids if all_input_ids is not None else input_ids
        new_caches = [] if use_cache else None
        total_aux_loss = None

        # Determine past_len if any kv_cache exists
        past_len = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0] is not None:
            c0 = kv_caches[0]
            if isinstance(c0, (tuple, list)) and len(c0) > 0:
                first_item = c0[0]
                if isinstance(first_item, (tuple, list)):
                    past_len = first_item[0].shape[2]
                else:
                    past_len = first_item.shape[2]

        device_type = x.device.type
        if torch.is_autocast_enabled(device_type):
            rope_dtype = torch.get_autocast_dtype(device_type)
        else:
            rope_dtype = target_dtype
        rope_cos, rope_sin = self.rope(rope_dtype, T, offset=past_len)

        for idx, layer in enumerate(self.layers):
            layer_cache = kv_caches[idx] if kv_caches is not None else None
            # Selective gradient checkpointing: checkpoint even layers and all engram-bearing layers
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled() and not use_cache:
                if idx % 2 == 0 or layer.has_engram:
                    x, layer_aux = torch.utils.checkpoint.checkpoint(
                        layer.forward_train, x, full_ids, rope_cos, rope_sin, use_reentrant=False
                    )
                else:
                    x, layer_aux = layer.forward_train(x, input_ids=full_ids, rope_cos=rope_cos, rope_sin=rope_sin)
                updated_cache = None
            else:
                x, updated_cache, layer_aux = layer(
                    x, input_ids=full_ids, kv_cache=layer_cache, use_cache=use_cache, rope_cos=rope_cos, rope_sin=rope_sin
                )
            if isinstance(layer_aux, torch.Tensor) and layer_aux.requires_grad:
                total_aux_loss = layer_aux if total_aux_loss is None else (total_aux_loss + layer_aux)
            if use_cache:
                new_caches.append(updated_cache)

            # Reasoning-After-Retrieval: Inject 80M Global Engram after Layer index 16 (17th layer),
            # leaving Layers 17..23 (7 layers) for deep reasoning!
            if idx == 16:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled() and not use_cache:
                    x, global_aux = torch.utils.checkpoint.checkpoint(
                        self.global_engram, x, full_ids, use_reentrant=False
                    )
                else:
                    x, global_aux = self.global_engram(x, input_ids=full_ids)
                if isinstance(global_aux, torch.Tensor) and global_aux.requires_grad:
                    total_aux_loss = global_aux if total_aux_loss is None else (total_aux_loss + global_aux)

        if total_aux_loss is None:
            total_aux_loss = self.zero_loss

        # Final RMSNorm
        hidden_states = self.final_norm(x)

        # Output LM Head in fp32 (tied weight is already fp32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(hidden_states.float(), self.lm_head.weight.float())

        return logits, new_caches, hidden_states, total_aux_loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.05,
        repetition_penalty: float = 1.15,
        repetition_window: int = 64,
        no_repeat_ngram_size: int = 3,
        eos_token_id: int = 2,
        stop_token_ids: Optional[List[int]] = None,
        suppress_raw_bytes: bool = False,
    ) -> torch.Tensor:
        """
        Auto-regressive generation with clean KV caching, sliding-window repetition penalty,
        no-repeat n-gram blocking, and min-p sampling to eliminate repetitive phrase loops.
        """
        self.eval()
        device = prompt_ids.device
        curr_ids = prompt_ids.clone()
        B = curr_ids.shape[0]
        all_tokens = [curr_ids[b].tolist() for b in range(B)]

        stop_set = set(stop_token_ids or [])
        stop_set.add(eos_token_id)

        autocast_device = "cuda" if "cuda" in str(device) else "cpu"
        with torch.amp.autocast(autocast_device, dtype=torch.bfloat16):
            # Prefill prompt with fresh KV cache
            logits, kv_caches, _, _ = self.forward(curr_ids, use_cache=True)
            next_token_logits = logits[:, -1, :].clone()

            for _ in range(max_new_tokens):
                for b in range(B):
                    b_tokens = all_tokens[b]
                    # 1. No-repeat n-gram blocking
                    if no_repeat_ngram_size > 0 and len(b_tokens) >= no_repeat_ngram_size:
                        ngram_prefix = tuple(b_tokens[-(no_repeat_ngram_size - 1):])
                        banned_tokens = set()
                        for i in range(len(b_tokens) - no_repeat_ngram_size + 1):
                            if tuple(b_tokens[i : i + no_repeat_ngram_size - 1]) == ngram_prefix:
                                banned_tokens.add(b_tokens[i + no_repeat_ngram_size - 1])
                        for b_tok in banned_tokens:
                            next_token_logits[b, b_tok] = -float("Inf")

                    # 2. Sliding-window repetition penalty (vectorized)
                    if repetition_penalty != 1.0 and len(b_tokens) > 0:
                        window = b_tokens[-repetition_window:] if repetition_window > 0 else b_tokens
                        unique_window_tokens = list(set(window))
                        indices = torch.tensor(unique_window_tokens, dtype=torch.long, device=device)
                        vals = next_token_logits[b, indices]
                        neg_mask = vals < 0
                        vals[neg_mask] *= repetition_penalty
                        vals[~neg_mask] /= repetition_penalty
                        next_token_logits[b, indices] = vals

                # 3. Sampling
                if temperature <= 0.0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    scaled_logits = next_token_logits / max(temperature, 1e-5)

                    # 4. Min-P Filtering (filter out low probability noise)
                    if min_p > 0.0:
                        probs = F.softmax(scaled_logits, dim=-1)
                        max_prob = torch.max(probs, dim=-1, keepdim=True).values
                        min_thresh = min_p * max_prob
                        scaled_logits[probs < min_thresh] = -float("Inf")

                    # 5. Top-K Cutoff
                    if top_k > 0:
                        topk_vals, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)), dim=-1)
                        scaled_logits[scaled_logits < topk_vals[:, [-1]]] = -float("Inf")

                    # 6. Top-P Nucleus Cutoff
                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            -1, sorted_indices, sorted_indices_to_remove
                        )
                        scaled_logits[indices_to_remove] = -float("Inf")

                    probs = F.softmax(scaled_logits, dim=-1)
                    if torch.isnan(probs).any() or (probs.sum(dim=-1) == 0).any():
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)

                curr_ids = torch.cat([curr_ids, next_token], dim=1)
                for b in range(B):
                    all_tokens[b].append(next_token[b, 0].item())

                if B == 1 and next_token[0, 0].item() in stop_set:
                    break
                elif B > 1 and all(all_tokens[b][-1] in stop_set for b in range(B)):
                    break

                # Forward single step with KV cache
                logits, kv_caches, _, _ = self.forward(next_token, kv_caches=kv_caches, use_cache=True, all_input_ids=curr_ids)
                next_token_logits = logits[:, -1, :].clone()

        return curr_ids

    def count_parameters(self) -> Dict[str, int]:
        """Returns detailed parameter breakdown across dense backbone and engram tables."""
        total_params = sum(p.numel() for p in self.parameters())
        engram_params = self.global_engram.num_parameters + sum(
            layer.engram.num_parameters for layer in self.layers if layer.has_engram
        )
        dense_params = total_params - engram_params
        return {
            "total_parameters": total_params,
            "dense_backbone_total": dense_params,
            "engram_tables_total": engram_params,
        }

