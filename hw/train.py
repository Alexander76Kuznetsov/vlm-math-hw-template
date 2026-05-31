from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any
from functools import partial

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    loss = out["loss"] if isinstance(out, dict) else out.loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def _torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "fp16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}.get(str(name), torch.float32)


def _save_adapter(model: MathVLM, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.adapter.state_dict()
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(state, str(path))
    else:
        torch.save(state, path)


def collate_fn(batch, processor):
    if len(batch) == 1 and isinstance(batch[0], list):
        batch = batch[0]
    return processor.collate([processor(sample) for sample in batch])


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point.

    TODO:
        - instantiate dataset, processor, model;
        - create DataLoader;
        - support max_steps and fast_train;
        - save adapter/checkpoint if configured.
    """
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    proc_cfg = config.get("processor", {})
    trainer_cfg = config.get("trainer", {})


    device = torch.device(trainer_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(trainer_cfg.get("dtype", "float32"))

    dataset = MathVQADataset(
        data_cfg.get("train_manifest", "assets/toy_math_vqa/manifest.jsonl"),
        split=data_cfg.get("split", "train"),
        max_samples=8 if fast_train else data_cfg.get("max_samples"),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["language_model"], trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN, IMAGE_START_TOKEN, IMAGE_END_TOKEN]})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision = AutoModel.from_pretrained(model_cfg["vision_encoder"], torch_dtype=dtype, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(model_cfg["language_model"], torch_dtype=dtype, trust_remote_code=True)
    lm.resize_token_embeddings(len(tokenizer))

    processor = MathVLMProcessor(tokenizer, ProcessorConfig(**proc_cfg))
    cfg = ModelConfig(
        vision_hidden_size=int(getattr(vision.config, "hidden_size")),
        text_hidden_size=int(lm.get_input_embeddings().embedding_dim),
        num_image_tokens=int(proc_cfg.get("num_image_tokens", 49)),
        image_token_id=int(tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)),
    )
    model = MathVLM(vision, lm, cfg).to(device)
    if model_cfg.get("freeze_vision", True) or model_cfg.get("freeze_llm", True):
        model.freeze_backbones()
    model.adapter.train()

    batch_size = int(trainer_cfg.get("local_batch_size", 1))
    num_workers = int(trainer_cfg.get("num_workers", 0))
    global_batch_size = int(trainer_cfg.get("global_batch_size", batch_size))
    grad_accum = max(1, global_batch_size // batch_size)
    max_steps = int(trainer_cfg.get("max_steps", 10))
    if fast_train:
        max_steps = min(max_steps, 2)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, processor=processor),
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(trainer_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=max_steps, desc="adapter-only train")
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss = out.loss if hasattr(out, "loss") else out[0]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {loss.item()}")
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
            if step >= max_steps:
                break
    pbar.close()

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        _save_adapter(model, save_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
