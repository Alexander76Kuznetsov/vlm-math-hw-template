from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from hw.constants import CHOICES, IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig


def normalize_text(s: str) -> str:
    return str(s).strip().lower().replace(",", ".")


def latin_letter(ch: str) -> str:
    mapping = {
        "А": "A",
        "В": "B",
        "С": "C",
        "Д": "D",
        "а": "A",
        "в": "B",
        "с": "C",
        "д": "D",
    }
    return mapping.get(ch, ch.upper())


def parse_mc_answer(text: str, options=None) -> str | None:
    """Extract multiple-choice answer letter from model output.

    TODO:
        Handle cases like:
            "A"
            "(B)"
            "Answer: C"
            "The correct answer is D."
    """
    text = str(text).strip()

    # 1. Явное "Ответ: B" / "Answer: C"
    m = re.search(r"(?:ответ|answer)\s*[:\-]?\s*([ABCDАВСД])\b", text, re.IGNORECASE)
    if m:
        return latin_letter(m.group(1))

    # 2. Формат "B) ..." где угодно
    m = re.search(r"\b([ABCDАВСД])\s*[\)\.:\-]", text, re.IGNORECASE)
    if m:
        return latin_letter(m.group(1))

    # 3. Просто одиночная буква
    m = re.search(r"\b([ABCDАВСД])\b", text, re.IGNORECASE)
    if m:
        return latin_letter(m.group(1))

    # 4. Если модель вывела значение, например "12" или "135°",
    # пробуем сопоставить с текстом вариантов.
    if options:
        out_norm = normalize_text(text)
        for i, opt in enumerate(options):
            letter = "ABCD"[i]
            opt_norm = normalize_text(opt)

            # убираем "A)", "B.", etc.
            opt_value = re.sub(r"^[abcdавсд]\s*[\)\.:\-]\s*", "", opt_norm).strip()

            if opt_value and opt_value in out_norm:
                return letter

            # числовое сравнение
            nums_out = re.findall(r"-?\d+(?:\.\d+)?", out_norm)
            nums_opt = re.findall(r"-?\d+(?:\.\d+)?", opt_value)
            if nums_out and nums_opt and nums_out[0] == nums_opt[0]:
                return letter

    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def _torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}.get(str(name), torch.float32)


def _load_adapter(model: MathVLM, path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu")
    if any(k.startswith("adapter.") for k in state):
        state = {k.removeprefix("adapter."): v for k, v in state.items() if k.startswith("adapter.")}
    model.adapter.load_state_dict(state, strict=False)


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop.

    TODO:
        - load eval dataset;
        - build prompts;
        - call model.generate;
        - parse answers;
        - write predictions if output_path is provided;
        - return metrics.
    """
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    proc_cfg = config.get("processor", {})
    inf_cfg = config.get("inference", {})

    manifest = data_cfg.get("eval_manifest", "assets/toy_math_vqa/manifest.jsonl")
    split = data_cfg.get("split", "dev")
    dataset = MathVQADataset(manifest, split=split, max_samples=data_cfg.get("max_samples"))

    device = torch.device(inf_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(inf_cfg.get("dtype", "float32"))

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
    model.freeze_backbones()
    if model_cfg.get("adapter_path"):
        _load_adapter(model, model_cfg["adapter_path"])
    model.eval()

    rows: list[dict[str, Any]] = []
    for sample in dataset:
        batch = processor.collate([processor(sample)])
        batch = {k: v.to(device) for k, v in batch.items()}
        gen = model.generate(
            batch,
            max_new_tokens=int(inf_cfg.get("max_new_tokens", 16)),
            do_sample=bool(inf_cfg.get("do_sample", False)),
            temperature=float(inf_cfg.get("temperature", 1.0)) if inf_cfg.get("do_sample", False) else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        pred = parse_mc_answer(text, sample.options) or normalize_text(text)
        rows.append({"id": sample.id, "prediction": pred, "answer": sample.answer, "subject": sample.subject, "output": text})

    output_path = inf_cfg.get("output_path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
