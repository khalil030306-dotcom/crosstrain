
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """
    Split out_features across tp ranks. Optionally all_gather outputs.
    """
    def __init__(self, in_features, out_features, bias=True,
                 tp_group=None, tp_rank=0, tp_size=1, gather_output=False, dtype=None, device=None):
        super().__init__()
        assert out_features % tp_size == 0, \
            f"out_features {out_features} not divisible by tp_size {tp_size}"
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group if tp_group is not None else dist.group.WORLD
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.out_per_rank = out_features // tp_size
        self.gather_output = gather_output
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((self.out_per_rank, in_features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(self.out_per_rank, **factory_kwargs)) if bias else None
        # init done later by shard_from_linear

    @torch.no_grad()
    def shard_from_linear(self, linear: nn.Linear):
        # copy slice of weight/bias from a full Linear
        w = linear.weight
        b = linear.bias
        start = self.tp_rank * self.out_per_rank
        end = (self.tp_rank + 1) * self.out_per_rank
        self.weight.copy_(w[start:end, :])
        if self.bias is not None:
            if b is not None:
                self.bias.copy_(b[start:end])
            else:
                nn.init.zeros_(self.bias)
        return self

    def forward(self, x):
        # [*, in] @ [out_per_rank, in]^T -> [*, out_per_rank]
        y_part = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y_part = y_part + self.bias
        if self.gather_output and self.tp_size > 1:
            parts = [torch.empty_like(y_part) for _ in range(self.tp_size)]
            dist.all_gather(parts, y_part, group=self.tp_group)
            y = torch.cat(parts, dim=-1)
            return y
        return y_part


class RowParallelLinear(nn.Module):
    """
    Split in_features across tp ranks. Sum across ranks at the end of forward.
    Expects its input to be the local slice along the last dim if column-partitioned
    by the previous layer; if given a full tensor, we slice internally.
    """
    def __init__(self, in_features, out_features, bias=True,
                 tp_group=None, tp_rank=0, tp_size=1, input_is_parallel=True, dtype=None, device=None):
        super().__init__()
        assert in_features % tp_size == 0, \
            f"in_features {in_features} not divisible by tp_size {tp_size}"
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group if tp_group is not None else dist.group.WORLD
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.input_is_parallel = input_is_parallel
        self.in_per_rank = in_features // tp_size
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, self.in_per_rank), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        # init done later by shard_from_linear

    @torch.no_grad()
    def shard_from_linear(self, linear: nn.Linear):
        w = linear.weight  # [out, in]
        b = linear.bias
        start = self.tp_rank * self.in_per_rank
        end = (self.tp_rank + 1) * self.in_per_rank
        self.weight.copy_(w[:, start:end])
        if self.bias is not None:
            if b is not None:
                self.bias.copy_(b)
            else:
                nn.init.zeros_(self.bias)
        return self

    def forward(self, x):
        # If we received a full tensor, slice the last dim
        if not self.input_is_parallel:
            start = self.tp_rank * self.in_per_rank
            end = (self.tp_rank + 1) * self.in_per_rank
            x = x[..., start:end]
        # [*, in_per_rank] @ [out, in_per_rank]^T = [*, out]
        y_local = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            y_local = y_local + self.bias
        if self.tp_size > 1:
            dist.all_reduce(y_local, op=dist.ReduceOp.SUM, group=self.tp_group)
        return y_local
