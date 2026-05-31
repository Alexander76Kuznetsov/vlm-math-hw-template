from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens
        hidden = max(text_hidden_size, vision_hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, text_hidden_size),
        )

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        if vision_hidden_states.ndim != 3:
            raise ValueError("vision_hidden_states must have shape [B, N, C]")
        x = vision_hidden_states[:, 1:, :] if vision_hidden_states.shape[1] > 1 else vision_hidden_states
        if x.shape[1] != self.num_image_tokens:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_image_tokens).transpose(1, 2)
        return self.proj(x)


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    result = input_embeds.clone()
    for b in range(input_ids.shape[0]):
        positions = torch.nonzero(input_ids[b] == image_token_id, as_tuple=False).flatten()
        if positions.numel() != visual_embeds.shape[1]:
            raise ValueError(
                f"Sample {b} has {positions.numel()} image tokens, expected {visual_embeds.shape[1]}"
            )
        result[b, positions, :] = visual_embeds[b].to(result.dtype)
    return result


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def _encode_visual(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, t = pixel_values.shape[:2]
        flat = pixel_values.view(b * t, *pixel_values.shape[2:])
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=flat)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)):
            hidden = outputs[0]
        if hidden is None:
            raise ValueError("vision encoder must return last_hidden_state")
        adapter_dtype = next(self.adapter.parameters()).dtype
        hidden = hidden.to(dtype=adapter_dtype)
        visual = self.adapter(hidden)
        return visual.view(b, t, self.config.num_image_tokens, self.config.text_hidden_size).mean(dim=1)

    def _inputs_embeds(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        visual_embeds = self._encode_visual(batch["pixel_values"])
        text_embeds = self.language_model.get_input_embeddings()(batch["input_ids"])
        return merge_visual_embeddings(text_embeds, batch["input_ids"], visual_embeds, self.config.image_token_id)

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        inputs_embeds = self._inputs_embeds(batch)
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        inputs_embeds = self._inputs_embeds(batch)
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            **generation_kwargs,
        )
