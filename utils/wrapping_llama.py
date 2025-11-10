
import torch
import torch.nn as nn

from .tp_layers import ColumnParallelLinear, RowParallelLinear

LLAMA_ATTN_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
LLAMA_MLP_NAMES = ("gate_proj", "up_proj", "down_proj")

def _maybe_bias(module: nn.Linear) -> bool:
    return module.bias is not None

@torch.no_grad()
def apply_llama_tp(model: nn.Module, tp_group, tp_rank: int, tp_size: int, dtype=None, device=None):
    """
    Replace select Linear layers in a HuggingFace LlamaForCausalLM with TP-aware counterparts.

    Design:
      - attention q/k/v: ColumnParallelLinear with gather_output=True (keep downstream code unchanged)
      - attention o_proj: RowParallelLinear (sum across ranks)
      - mlp gate/up: ColumnParallelLinear (keep split), gather_output=False
      - mlp down: RowParallelLinear (sum)
      - keep lm_head as full (no sharding) for simplicity

    Args:
      model: LlamaForCausalLM
    """
    # 1. Iterate transformer blocks
    for i, blk in enumerate(model.model.layers):
        attn = blk.self_attn
        mlp = blk.mlp

        # --- Attention ---
        # q, k, v: column parallel (gather to full to avoid changing head logic)
        for name in ("q_proj", "k_proj", "v_proj"):
            lin: nn.Linear = getattr(attn, name)
            bias = _maybe_bias(lin)
            col = ColumnParallelLinear(lin.in_features, lin.out_features, bias=bias,
                                       tp_group=tp_group, tp_rank=tp_rank, tp_size=tp_size,
                                       gather_output=True, dtype=lin.weight.dtype, device=lin.weight.device)
            col.shard_from_linear(lin)
            setattr(attn, name, col)

        # o_proj: row parallel (sum-reduce)
        lin_o: nn.Linear = getattr(attn, "o_proj")
        bias_o = _maybe_bias(lin_o)
        row = RowParallelLinear(lin_o.in_features, lin_o.out_features, bias=bias_o,
                                tp_group=tp_group, tp_rank=tp_rank, tp_size=tp_size,
                                input_is_parallel=True, dtype=lin_o.weight.dtype, device=lin_o.weight.device)
        row.shard_from_linear(lin_o)
        setattr(attn, "o_proj", row)

        # --- MLP ---
        # gate_proj, up_proj: column-parallel (no gather)
        for name in ("gate_proj", "up_proj"):
            lin: nn.Linear = getattr(mlp, name)
            bias = _maybe_bias(lin)
            col = ColumnParallelLinear(lin.in_features, lin.out_features, bias=bias,
                                       tp_group=tp_group, tp_rank=tp_rank, tp_size=tp_size,
                                       gather_output=False, dtype=lin.weight.dtype, device=lin.weight.device)
            col.shard_from_linear(lin)
            setattr(mlp, name, col)

        # down_proj: row-parallel
        lin_d: nn.Linear = getattr(mlp, "down_proj")
        bias_d = _maybe_bias(lin_d)
        row_d = RowParallelLinear(lin_d.in_features, lin_d.out_features, bias=bias_d,
                                  tp_group=tp_group, tp_rank=tp_rank, tp_size=tp_size,
                                  input_is_parallel=True, dtype=lin_d.weight.dtype, device=lin_d.weight.device)
        row_d.shard_from_linear(lin_d)
        setattr(mlp, "down_proj", row_d)

    return model
