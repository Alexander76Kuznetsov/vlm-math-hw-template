from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None) or "<|pad|>"

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image.thumbnail((self.config.image_size, self.config.image_size), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.config.image_size, self.config.image_size), (255, 255, 255))
        left = (self.config.image_size - image.width) // 2
        top = (self.config.image_size - image.height) // 2
        canvas.paste(image, (left, top))
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(canvas.tobytes()))
        tensor = data.view(self.config.image_size, self.config.image_size, 3).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        tensor = self._image_to_tensor(image)
        if self.config.num_tiles <= 1:
            return tensor.unsqueeze(0)

        n = int(self.config.num_tiles)
        cols = int(torch.ceil(torch.sqrt(torch.tensor(float(n)))).item())
        rows = int(torch.ceil(torch.tensor(float(n)) / cols).item())
        tiles: list[torch.Tensor] = []
        _, h, w = tensor.shape
        for i in range(n):
            r, c = divmod(i, cols)
            y0 = round(r * h / rows)
            y1 = round((r + 1) * h / rows)
            x0 = round(c * w / cols)
            x1 = round((c + 1) * w / cols)
            tile = tensor[:, y0:y1, x0:x1].unsqueeze(0)
            tile = F.interpolate(tile, size=(self.config.image_size, self.config.image_size), mode="bilinear", align_corners=False)
            tiles.append(tile.squeeze(0))
        return torch.stack(tiles, dim=0)

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        image_tokens = IMAGE_START_TOKEN + (IMAGE_TOKEN * self.config.num_image_tokens) + IMAGE_END_TOKEN
        options = "\n".join(sample.options)
        prompt = (
            f"{image_tokens}\n"
            "Реши визуально-математическую задачу. "
            "Выбери один вариант ответа и напиши только букву.\n\n"
            f"Вопрос: {sample.question}\n"
        )
        if options:
            prompt += f"Варианты:\n{options}\n"
        prompt += "Ответ:"
        if include_answer:
            prompt += f" {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        prompt = self.build_prompt(sample, include_answer=False)
        full = self.build_prompt(sample, include_answer=True)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.config.max_length)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=True, truncation=True, max_length=self.config.max_length)["input_ids"]
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[: min(len(prompt_ids), len(labels))] = self.config.ignore_index
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        max_len = max(x["input_ids"].numel() for x in batch)
        out: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in batch:
            pad = max_len - item["input_ids"].numel()
            out["input_ids"].append(F.pad(item["input_ids"], (0, pad), value=pad_id))
            out["attention_mask"].append(F.pad(item["attention_mask"], (0, pad), value=0))
            out["labels"].append(F.pad(item["labels"], (0, pad), value=self.config.ignore_index))
        return {
            "input_ids": torch.stack(out["input_ids"]),
            "attention_mask": torch.stack(out["attention_mask"]),
            "labels": torch.stack(out["labels"]),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        }
