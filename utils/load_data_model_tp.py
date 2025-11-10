import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, DataCollatorForLanguageModeling


# --- NEW: Optional TP sharding helper ---
# utils/load_data_model_tp.py 里替换/更新这个函数
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor.parallel import parallelize_module
from transformers.models.llama.modeling_llama import LlamaAttention

def debug_parallelize_layerwise(model, tp_mesh, plan: dict, tp_group, logger=None):
    """
    逐层应用 TP：按层路径字符串有序地，一个一个调用 parallelize_module。
    哪层出错能立刻定位到。
    """
    # 确保所有 rank 对 plan 的遍历顺序完全一致
    keys = sorted(plan.keys())

    for k in keys:
        single = {k: plan[k]}
        if logger:
            logger.info(f"[TP-layerwise] applying {k} -> {type(plan[k]).__name__}")
        if tp_group is not None:
            dist.barrier(group=tp_group)  # 层前同步，消除时序漂移
        try:
            # 只对这一层并行化
            parallelize_module(model, tp_mesh, single)
        except Exception as e:
            # 打印该层的维度信息帮助判断整除/偏置问题
            mod = model.get_submodule(k)
            shape = (getattr(mod, "in_features", None), getattr(mod, "out_features", None))
            has_bias = getattr(mod, "bias", None) is not None
            msg = f"TP failed at layer '{k}', shape={shape}, bias={has_bias}: {repr(e)}"
            if logger: logger.error(msg)
            raise
        finally:
            if tp_group is not None:
                dist.barrier(group=tp_group)  # 层后同步
    if logger:
        logger.info(f"[TP-layerwise] all {len(keys)} layers applied successfully.")
    return model

def apply_tp_sharding(model, tp_group, tp_size, logger=None):
    """
    Tensor Parallel sharding with DTensor.
    - 使用 DeviceMesh(一维) + parallelize_module
    - plan 的 key 必须是模块路径字符串（相对 `model` 的命名路径），不能是模块对象
    """
    tp_size = int(tp_size) if tp_size is not None else 1
    if tp_size <= 1:
        if logger: logger.info("TP disabled (tp_size<=1); skip sharding.")
        return model

    # 兼容不同 PyTorch 版本的导入路径
    try:
        from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
    except Exception:
        from torch.distributed._tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
    try:
        from torch.distributed._tensor import DeviceMesh
    except Exception:
        from torch.distributed.tensor import DeviceMesh

    if not dist.is_initialized():
        raise RuntimeError("Distributed not initialized before TP sharding.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % tp_size == 0, f"world_size={world_size} must be divisible by tp_size={tp_size}"
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size

    # 该 DP 切片内的 TP ranks（若你的 rank 映射不是连续，请按实际拓扑生成）
    tp_ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))

    # 构造 1D DeviceMesh（不要传 pg）
    tp_mesh = None
    try:
        tp_mesh = DeviceMesh(device_type="cuda",
                             mesh=torch.tensor(tp_ranks, dtype=torch.int64),
                             mesh_dim_names=("tp",))
    except TypeError:
        try:
            tp_mesh = DeviceMesh("cuda", tp_ranks)
        except TypeError:
            tp_mesh = DeviceMesh(torch.device("cuda"), tp_ranks)
    if tp_mesh is None:
        raise RuntimeError("Failed to construct DeviceMesh for TP.")

    # ---- 构造 plan：key 用命名路径字符串，而非模块对象 ----
    plan = {}
    matched_names = []
    for name, mod in model.named_modules():
        if not name:   # 跳过根模块空名
            continue
        if isinstance(mod, nn.Linear):
            # LLaMA/Qwen 习惯：q/k/v/gate/up 列并行；o/down 行并行
            if any(k in name for k in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
                plan[name] = ColwiseParallel()   # 切 out_features
                matched_names.append(name)
            elif any(k in name for k in ("o_proj", "down_proj")):
                plan[name] = RowwiseParallel()   # 切 in_features
                matched_names.append(name)

    if not plan:
        if logger: logger.warning("TP requested, but no Linear layers matched plan; skip TP.")
        return model

    # 维度可整除性预检查（仍可在此用 mod，因为还在循环体里拿得到它）
    not_divisible = []
    for name, mod in model.named_modules():
        if name in plan:
            spec = plan[name]
            if isinstance(spec, ColwiseParallel) and (mod.out_features % tp_size != 0):
                not_divisible.append(f"{name}.out_features={mod.out_features} !% {tp_size}")
            if isinstance(spec, RowwiseParallel) and (mod.in_features % tp_size != 0):
                not_divisible.append(f"{name}.in_features={mod.in_features} !% {tp_size}")
    if not_divisible and logger:
        logger.warning("Some Linear dims not divisible by tp_size; may error at sharding: " +
                       "; ".join(not_divisible[:6]))

    if logger:
        preview = ", ".join(matched_names[:6]) + (" ..." if len(matched_names) > 6 else "")
        logger.info(f"Applying TP: tp_size={tp_size}, tp_ranks={tp_ranks}, matched_layers={len(matched_names)} [{preview}]")

    # ---- 调用顺序： (module, device_mesh, parallelize_plan) ----
    # model = parallelize_module(model, tp_mesh, plan)
    model = debug_parallelize_layerwise(model, tp_mesh, plan, tp_group, logger)

    if logger: logger.info("TP sharding done.")
    return model

def adjust_attention_heads_for_tp(model, tp_size, logger):
    for name, module in model.named_modules():
        if hasattr(module, "num_heads") and hasattr(module, "head_dim"):
            if "self_attn" in name or isinstance(module, LlamaAttention):
                if module.num_heads % tp_size == 0:
                    old_heads = module.num_heads
                    module.num_heads = module.num_heads // tp_size
                    logger.info(f"[TP adjust] {name}: num_heads {old_heads} -> {module.num_heads}")
                if hasattr(module, "num_key_value_heads") and module.num_key_value_heads % tp_size == 0:
                    old_kv = module.num_key_value_heads
                    module.num_key_value_heads //= tp_size
                    logger.info(f"[TP adjust] {name}: num_kv_heads {old_kv} -> {module.num_key_value_heads}")



def load_sst2_data(tokenizer, batch_size, eval_batch_size, rank, world_size):
    """Load SST-2 dataset for sentiment classification"""
    train_dataset = load_dataset("glue", "sst2", split="train")
    eval_dataset = load_dataset("glue", "sst2", split="validation")
    
    block_size = 128
    
    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=block_size)
        
    # Preprocess train and validation sets
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Create distributed samplers and dataloaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
    
    return train_dataloader, eval_dataloader

def load_c4_data(tokenizer, batch_size, eval_batch_size, rank, world_size):
    """Load C4-en dataset for language modeling"""
    # Load only training data, no validation set
    train_files = [
        f"en/c4-train.{i:05d}-of-01024.json.gz"
        for i in range(256)
    ]
    ds = load_dataset("/data1/hfhub/c4", streaming=True,
                      data_files={
                "train": train_files,
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            })
    ds = ds.shuffle(seed=2025)
    
    block_size = 1024   # DiLoCo 论文
    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=block_size)
        return outputs

    tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"]).with_format("torch")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataset = split_dataset_by_node(tokenized_datasets["train"], world_size=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=batch_size, pin_memory = True, num_workers = 4)

    eval_dataloader = DataLoader(
        dataset=tokenized_datasets["validation"],
        collate_fn=data_collator,
        batch_size=eval_batch_size,
        pin_memory = True,
        num_workers = 4
    )

    # Return train dataloader and None for eval_dataloader, as we'll compute ppl differently
    return train_dataloader, eval_dataloader

def load_data_and_model(dataset_name, model_name, batch_size, eval_batch_size, rank, world_size):
    """
    Load dataset and model.
    Return Trainloader, Evalloader, tokenizer, model objects.
    """
    if dataset_name == "sst2" and model_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        train_dataloader, eval_dataloader = load_sst2_data(tokenizer, batch_size, eval_batch_size, rank, world_size)
        return train_dataloader, eval_dataloader, tokenizer, model, 'classification'
        
    elif dataset_name == "c4en":
        if model_name == "llama150m":
            model_name = "PrimeIntellect/llama-150m-fresh"
        elif model_name == 'llama1b':
            model_name = "/data1/hfhub/llama-1b-fresh"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = "</s>"  # Ensure pad token is set for models that need it
        model = AutoModelForCausalLM.from_pretrained(model_name)
        train_dataloader, eval_dataloader = load_c4_data(tokenizer, batch_size, eval_batch_size, rank, world_size)
        return train_dataloader, eval_dataloader, tokenizer, model, 'language_modeling'
        
    else:
        raise ValueError(f"Unsupported combination of dataset and model: {dataset_name}, {model_name}")
    
