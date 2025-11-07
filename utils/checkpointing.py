import os
import time
import glob
import torch
import torch.distributed as dist
from pathlib import Path
import logging
import re
import shutil
# import wandb
from typing import Dict, Any, Optional, Tuple, Union

# --- Helper Functions (No changes needed here) ---

def get_latest_checkpoint(checkpoint_dir: str, logger: Optional[logging.Logger] = None, rank: int = 0) -> str:
    """
    获取指定目录中最新的检查点目录和对应rank的文件路径。
    """
    if not os.path.exists(checkpoint_dir):
        if logger:
            logger.warning(f"检查点目录不存在: {checkpoint_dir}")
        return ""
    
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    if not checkpoint_dirs:
        if logger:
            logger.info(f"在目录 {checkpoint_dir} 中未找到检查点目录")
        return ""
    
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = checkpoint_dirs[0]
    
    rank_file = os.path.join(latest_dir, f"rank_{rank}.pt")
    if not os.path.exists(rank_file):
        if logger:
            logger.info(f"未找到rank {rank}的检查点文件: {rank_file}")
        return ""
    
    if logger:
        logger.info(f"找到最新检查点: {rank_file}")
    
    return rank_file


def _clean_old_checkpoints(checkpoint_dir: str, max_keep: int, logger: Optional[logging.Logger] = None) -> None:
    """
    清理旧的检查点目录，只保留最近的N个。
    """
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    if len(checkpoint_dirs) <= max_keep:
        return
    
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
    
    for old_dir in checkpoint_dirs[max_keep:]:
        if logger:
            logger.info(f"删除旧检查点目录: {old_dir}")
        shutil.rmtree(old_dir)


def _serialize_shard_tracker(
    shard_tracker: Dict[int, Dict[str, Any]],
    save_model_only: bool,
    include_old_sent: bool
) -> Dict[int, Dict[str, Any]]:
    serializable = {}
    for idx, info in shard_tracker.items():
        shard = {
            "sent_at_step": info["sent_at_step"],
            "next_receive_step": info["next_receive_step"],
            **({"old_sent_at_step": info["old_sent_at_step"]} if include_old_sent else {}),
            "params": [p.data.cpu().clone() for p in info["params"]],
            "staged_params": None if info["staged_params"] is None
                             else [p.data.cpu().clone() for p in info["staged_params"]],
        }
        if info.get("global_num_params") is not None:
            shard["global_num_params"] = info["global_num_params"]
        if info.get("global_num_bytes") is not None:
            shard["global_num_bytes"] = info["global_num_bytes"]
        if info.get("outer_optimizer") is not None and not save_model_only:
            shard["outer_optimizer_state_dict"] = info["outer_optimizer"].state_dict()
        serializable[idx] = shard
    return serializable


def _restore_shard_tracker(
    ckpt_shards: Dict[int, Dict[str, Any]],
    shard_tracker: Dict[int, Dict[str, Any]],
    include_old_sent: bool
) -> None:
    device = next(iter(shard_tracker.values()))["params"][0].device
    for idx, ck in ckpt_shards.items():
        if idx not in shard_tracker: continue
        tgt = shard_tracker[idx]
        tgt["sent_at_step"] = ck["sent_at_step"]
        tgt["next_receive_step"] = ck["next_receive_step"]
        if ck.get("global_num_params") is not None:
            tgt["global_num_params"] = ck["global_num_params"]
        if ck.get("global_num_bytes") is not None:
            tgt["global_num_bytes"] = ck["global_num_bytes"]
        if include_old_sent:
            tgt["old_sent_at_step"] = ck["old_sent_at_step"]
        for i, cpu_t in enumerate(ck["params"]):
            tgt["params"][i].data.copy_(cpu_t.to(device))
        if ck["staged_params"] is not None:
            if tgt["staged_params"] is None:
                tgt["staged_params"] = [p.to(device).clone() for p in ck["staged_params"]]
            else:
                for i, cpu_t in enumerate(ck["staged_params"]):
                    tgt["staged_params"][i].data.copy_(cpu_t.to(device))
        if "outer_optimizer_state_dict" in ck and tgt.get("outer_optimizer") is not None:
            tgt["outer_optimizer"].load_state_dict(ck["outer_optimizer_state_dict"])




def save_checkpoint(
    algorithm: str,
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    original_snapshot: Optional[torch.nn.Module] = None,
    outer_optimizer: Optional[torch.optim.Optimizer] = None,
    shard_tracker: Optional[Dict[int, Dict[str, Any]]] = None,
    epoch: int = 0,
    global_step: int = 0,
    micro_step: int = 0,
    comp_time_total: float = 0.0,
    comm_time_total: float = 0.0,
    comm_vol_total: float = 0.0,
    metric_value: Optional[float] = None,
    is_best: bool = False,
    rank: int = 0,
    save_model_only: bool = False,
    max_checkpoints: int = 3,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    统一的检查点保存函数，支持 'diloco', 'streaming', 'dc' 算法。
    """
    # 1. 通用逻辑：创建目录和文件名
    os.makedirs(checkpoint_dir, exist_ok=True)
    step_dir_name = f"step_{global_step}"
    dist.barrier()
    step_dir = os.path.join(checkpoint_dir, step_dir_name)
    os.makedirs(step_dir, exist_ok=True)
    checkpoint_file = f"rank_{rank}.pt"
    tmp_checkpoint_path = os.path.join(step_dir, f"tmp_{checkpoint_file}")
    checkpoint_path = os.path.join(step_dir, checkpoint_file)

    # 2. 准备通用的状态字典
    checkpoint = {
        "algorithm": algorithm,  # <-- 关键：保存算法类型
        "epoch": epoch,
        "global_step": global_step,
        "micro_step": micro_step,
        "comp_time_total": comp_time_total,
        "comm_time_total": comm_time_total,
        "comm_vol_total": comm_vol_total,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if not save_model_only else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler and not save_model_only else None,
        "scaler_state_dict": scaler.state_dict() if scaler and not save_model_only else None,
        "rank": rank,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }
    # if rank == 0 and wandb.run is not None:
    #     checkpoint["wandb_run_id"] = wandb.run.id

    # 3. 根据算法类型，添加特定的状态
    if algorithm == 'diloco':
        if original_snapshot is not None:
            checkpoint["original_snapshot_state_dict"] = original_snapshot.state_dict()
        if outer_optimizer is not None and not save_model_only:
            checkpoint["outer_optimizer_state_dict"] = outer_optimizer.state_dict()
    elif algorithm == 'streaming':
        if shard_tracker is not None:
            checkpoint["shard_tracker"] = _serialize_shard_tracker(shard_tracker, save_model_only, include_old_sent=False)
    elif algorithm == 'dc':
        if shard_tracker is not None:
            checkpoint["shard_tracker"] = _serialize_shard_tracker(shard_tracker, save_model_only, include_old_sent=True)
        
    else:
        raise ValueError(f"未知的算法类型: {algorithm}")

    # 4. 通用逻辑：执行保存
    if logger:
        logger.info(f"Rank {rank}: 正在为 '{algorithm}' 算法保存检查点到 {checkpoint_path}")
    torch.save(checkpoint, tmp_checkpoint_path)
    os.replace(tmp_checkpoint_path, checkpoint_path)

    # 5. 通用逻辑：处理最佳模型和清理旧检查点
    if is_best and rank == 0:
        best_dir = os.path.join(checkpoint_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        best_path = os.path.join(best_dir, "rank_0.pt")
        shutil.copy2(checkpoint_path, best_path)
        with open(os.path.join(checkpoint_dir, "best_step.txt"), "w") as f:
            f.write(f"step: {global_step}\nmetric: {metric_value if metric_value is not None else 'N/A'}\n")
        if logger:
            logger.info(f"Rank {rank}: 已保存最佳模型到 {best_path}")
    dist.barrier()
    if is_best and rank != 0:
        best_rank_path = os.path.join(checkpoint_dir, "best_model", f"rank_{rank}.pt")
        shutil.copy2(checkpoint_path, best_rank_path)

    if rank == 0 and max_checkpoints > 0:
        _clean_old_checkpoints(checkpoint_dir, max_checkpoints, logger)
    
    dist.barrier()
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    original_snapshot: Optional[torch.nn.Module] = None,
    outer_optimizer: Optional[torch.optim.Optimizer] = None,
    shard_tracker: Optional[Dict[int, Dict[str, Any]]] = None,
    rank: int = 0,
    map_location: str = "cpu",
    load_model_only: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    统一的检查点加载函数，能自动识别并加载 'diloco', 'streaming', 'dc' 算法的检查点。
    """
    # 1. 通用逻辑：查找并验证检查点文件
    checkpoint_file = get_latest_checkpoint(checkpoint_path, logger, rank) if os.path.isdir(checkpoint_path) else checkpoint_path
    if not os.path.exists(checkpoint_file):
        if logger:
            logger.error(f"检查点文件不存在: {checkpoint_file}")
        return {}
    
    if logger:
        logger.info(f"Rank {rank}: 正在加载检查点: {checkpoint_file}")
    
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    
    # 2. 关键：从文件中读取算法类型
    algorithm = checkpoint.get("algorithm")
    if logger:
        logger.info(f"检测到检查点算法类型: '{algorithm}'")

    # 3. 恢复通用状态
    model.load_state_dict(checkpoint["model_state_dict"])
    if not load_model_only:
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # 恢复随机数状态
    if "rng_states" in checkpoint:
        rng_states = checkpoint["rng_states"]
        if rng_states.get("torch") is not None:
            torch.set_rng_state(torch.tensor(rng_states["torch"], dtype=torch.uint8, device="cpu"))
        if torch.cuda.is_available() and rng_states.get("cuda") is not None:
            cuda_rng_list = [torch.tensor(s, dtype=torch.uint8, device="cpu") for s in rng_states["cuda"]]
            torch.cuda.set_rng_state_all(cuda_rng_list)

    # 4. 根据算法类型，恢复特定状态
    if algorithm == 'diloco':
        if original_snapshot and "original_snapshot_state_dict" in checkpoint:
            original_snapshot.load_state_dict(checkpoint["original_snapshot_state_dict"])
        if outer_optimizer and "outer_optimizer_state_dict" in checkpoint and not load_model_only:
            outer_optimizer.load_state_dict(checkpoint["outer_optimizer_state_dict"])
    elif algorithm == 'streaming':
        if shard_tracker and "shard_tracker" in checkpoint:
            _restore_shard_tracker(checkpoint["shard_tracker"], shard_tracker, include_old_sent=False)
    elif algorithm == 'dc':
        if shard_tracker and "shard_tracker" in checkpoint:
            _restore_shard_tracker(checkpoint["shard_tracker"], shard_tracker, include_old_sent=True)
    
    # 5. 提取并返回训练状态
    training_state = {k: checkpoint.get(k, 0) for k in ["epoch", "global_step", "micro_step"]}
    training_state.update({k: checkpoint.get(k, 0.0) for k in ["comp_time_total", "comm_time_total", "comm_vol_total"]})
    training_state["algorithm"] = algorithm # <-- 返回算法类型
    
    if rank == 0:
        training_state["wandb_run_id"] = checkpoint.get("wandb_run_id")

    if logger:
        logger.info(f"Rank {rank}: 恢复到轮次 {training_state['epoch']}, 步数 {training_state['global_step']}")
    
    dist.barrier()
    return training_state