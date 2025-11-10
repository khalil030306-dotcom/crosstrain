import torch.nn as nn
import torch
import torch.distributed as dist
import logging # 确保导入 logging
from typing import List, Tuple
def get_layer_shards(model: nn.Module, num_shards: int, pattern: str = "sequential"):
    """
    按层将 model.parameters() 划分成 num_shards 份：
      - sequential: 连续切分
      - stride: 跳跃切分
    返回 List[List[nn.Parameter]]，长度 = num_shards
    """
    # 1. 把每一“层”按顺序收集成一个大列表
    layers = []
    # embed
    layers.append(list(model.model.embed_tokens.parameters()))
    # transformer 层
    for lyr in model.model.layers:
        layers.append(list(lyr.parameters()))
    # norm + lm_head
    layers.append(list(model.model.norm.parameters()))
    layers.append(list(model.lm_head.parameters()))
    # 2. 根据 pattern 计算每个 shard 包含哪些 layer idx
    L = len(layers)
    shards = [[] for _ in range(num_shards)]
    if pattern == "sequential":
        per = (L + num_shards - 1) // num_shards
        for i, layer_params in enumerate(layers):
            shard_id = min(i // per, num_shards - 1)
            shards[shard_id].extend(layer_params)
    elif pattern == "stride":
        for i, layer_params in enumerate(layers):
            shard_id = i % num_shards
            shards[shard_id].extend(layer_params)
    else:
        raise ValueError(f"Unknown pattern {pattern}")
    return shards, L

#CHANGE 定义select_next_shard函数
def select_next_shard(shard_tracker, H, K, current_global_step, 
                      tp_group: dist.ProcessGroup,   # <-- 【A.1】新增 tp_group
                      dp_group: dist.ProcessGroup):  # <-- 【A.1】新增 dp_group
    """
    【已修正】:
    - 增加了 tp_group 和 dp_group 参数。
    - 1. 先在 TP 组内 all_reduce 范数，确保 TP 组内决策一致。
    - 2. 再在 DP 组内 all_reduce R值，确保 DP 组间决策一致。
    - 增加了对 group == None 的检查，以兼容 DP-only 模式。
    """
    
    # 1. 检查是否过度陈旧 (此逻辑不变)
    for shard_idx in range(K):
        if shard_idx not in shard_tracker: continue
        t_p_b = shard_tracker[shard_idx]["sent_at_step"]
        I_p = current_global_step - t_p_b
        if I_p >= H:
            return shard_idx # 优先选择过度陈旧的片段
            
    # 2. 计算所有分片的 R_p 值
    
    # 找到一个有效的 device
    device = None
    for info in shard_tracker.values():
        if info["params"]:
            device = info["params"][0].device
            break
    if device is None: device = torch.device("cpu")

    local_magnitudes = {} # 存储 {shard_idx: local_norm}
    
    for shard_idx in range(K):
        if shard_idx not in shard_tracker:
            continue
            
        if shard_tracker[shard_idx]["sent_at_step"] == 0:
            return shard_idx # 优先选择还没发送过的片段
            
        cur_shard = shard_tracker[shard_idx]
        
        # 2a. 计算【本地切片】的范数
        with torch.no_grad():
            # 注意：param_refs 是对模型当前参数的引用
            # staged_params 是旧的参数快照
            # 我们需要计算 (staged_params - current_params) 的范数
            # 【修正】应该是 (旧的全局快照 - 当前本地模型)
            # cur_shard["params"] = 旧的全局快照 (θ(t_rec - H)_m,p)
            # cur_shard["param_refs"] = 当前本地模型 (θ(t_now)_m,p)
            # 我们需要计算的是 (θ(t_rec - H)_m,p - θ(t_now)_m,p) 的范数
            sync_grads = [p_old.data - p_local.data
                          for p_old, p_local in zip(cur_shard["params"], cur_shard["param_refs"])]
        
        total_norm_sq = torch.tensor(0.0, device=device, dtype=torch.float32)
        with torch.no_grad():
            for tensor in sync_grads:
                total_norm_sq += tensor.norm(1, dtype=torch.float32) # L1 范数
        
        local_magnitudes[shard_idx] = total_norm_sq.item()
        del sync_grads

    # 2b. 【新】将在 TP 组内聚合这些范数
    #     创建一个包含所有【本地】范数的 tensor
    local_mag_tensor = torch.tensor(
        [local_magnitudes.get(i, 0.0) for i in range(K)], 
        device=device, 
        dtype=torch.float32
    )
    
    #     在 TP 组内 All-Reduce (SUM)
    # --- 【兼容性修改】 ---
    if tp_group is not None and dist.get_world_size(group=tp_group) > 1:
        dist.all_reduce(local_mag_tensor, op=dist.ReduceOp.SUM, group=tp_group)
    
    # 2c. 现在，所有 TP Ranks 都拥有了【全局（跨TP）】的范数
    #     `local_mag_tensor` 现在是 `global_mag_tensor`
    
    dict_R_values = {} # 存储 {shard_idx: global_R_value}
    for shard_idx in range(K):
        if shard_idx not in shard_tracker:
            continue
        
        # 使用聚合后的全局范数
        global_update_magnitude = local_mag_tensor[shard_idx].item() 
        
        cur_shard = shard_tracker[shard_idx]
        I_p = current_global_step - cur_shard["sent_at_step"]
        
        # 计算 R 值 (现在在 TP 组内的所有 Ranks 上都是一致的)
        current_R = 0.0
        if I_p > 0 and cur_shard["global_num_params"] > 0:
             # 乘以 10**8 以避免浮点数下溢
            current_R = global_update_magnitude * 10**8 / (I_p * cur_shard["global_num_params"])
        
        dict_R_values[shard_idx] = current_R

    # 3. 在 DP 组内聚合 R 值
    
    # 创建一个包含所有【全局 TP R 值】的 tensor
    dict_tensor = torch.tensor(
        [dict_R_values.get(i, -1.0) for i in range(K)], # 使用 -1.0 作为无效分片的占位符
        device=device, 
        dtype=torch.float32
    )
    
    # 在 DP 组上 All-Reduce (SUM)
    # --- 【兼容性修改】 ---
    # 如果 dp_group 为 None (tp_size=1)，则使用默认的 WORLD 组
    dp_group_to_use = dp_group if dp_group is not None else dist.group.WORLD
    dist.all_reduce(dict_tensor, op=dist.ReduceOp.SUM, group=dp_group_to_use)
    
    # 4. 选出索引 (现在所有 Ranks 上的结果都是一致的)
    selected_idx = torch.argmax(dict_tensor).item()
    
    del dict_tensor, local_mag_tensor
    return selected_idx


def print_shard_param_counts_from_shards(
    shards: list, 
    logger: logging.Logger, 
    tp_group: dist.ProcessGroup, # <-- 【A.1】新增 tp_group
    tp_rank: int,                # <-- 【A.1】新增 tp_rank
    dp_rank: int,                 # <-- 【A.1】新增 dp_rank
    device: torch.device
) -> Tuple[List[int], List[int]]:
    """
    【已修改】:
    - 增加了 tp_group, tp_rank, dp_rank 参数。
    - 在 TP 组内进行 All-Reduce 来获取“全局”（跨TP）的参数量。
    - 仅在全局 Rank 0 (dp_rank 0 + tp_rank 0) 上打印。
    - 增加了对 group == None 的检查，以兼容 DP-only 模式。
    
    shards: List[List[nn.Parameter]] (本地参数切片)
    作用：对每个 shard 统计参数个数、字节数；在 TP 组内做 SUM，打印全局统计
    """
    if not shards or not shards[0]:
        # 如果分片为空（例如在某些rank上），则返回空列表
        if dp_rank == 0 and tp_rank == 0:
            logger.warning("print_shard_param_counts_from_shards 接收到空的分片列表。")
        return [], []
        
    # 用第一个参数的 device，兼容 CUDA/NPU
    # dev = shards[0][0].device if shards[0] else torch.device("cpu")
    dev = device

    global_counts = []
    global_bytes  = []

    for shard in shards:
        # 本 rank 上该 shard 的参数个数/字节
        n_local = sum(p.numel() for p in shard)
        b_local = sum(p.numel() * p.element_size() for p in shard)

        n_t = torch.tensor([n_local], device=dev, dtype=torch.long)
        b_t = torch.tensor([b_local], device=dev, dtype=torch.long)

        # 【A.2】在 TP 组内做 SUM 得到“全局（跨 TP 切分）”的数值
        # --- 【兼容性修改】 ---
        if tp_group is not None and dist.get_world_size(group=tp_group) > 1:
            dist.all_reduce(n_t, op=dist.ReduceOp.SUM, group=tp_group)
            dist.all_reduce(b_t, op=dist.ReduceOp.SUM, group=tp_group)

        global_counts.append(int(n_t.item()))
        global_bytes.append(int(b_t.item()))

    # 【A.3】只在全局 Rank 0 (DP=0, TP=0) 打印一次，避免刷屏
    if dp_rank == 0 and tp_rank == 0:
        # --- 【兼容性修改】 ---
        tp_size = dist.get_world_size(group=tp_group) if tp_group is not None else 1
        head = f"[Shard Stats] (Global across TP_Size={tp_size})"
        logger.info(head)
        total_n = 0
        total_b = 0
        for i, (n, b) in enumerate(zip(global_counts, global_bytes)):
            mb = b / (1024**2)
            total_n += n
            total_b += b
            logger.info(f"  - Shard {i:02d}: params={n:<12,} \t bytes={mb:,.2f} MB")
        logger.info("-" * 40)
        logger.info(f"    Total: params={total_n:<12,} \t bytes={total_b / (1024**2):,.2f} MB")
            
    return global_counts, global_bytes