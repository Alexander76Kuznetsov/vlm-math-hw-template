from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MathVQASample:
    """One visual-math QA example.

    Attributes:
        id: Stable example id.
        image: PIL image in RGB mode.
        question: User question without hidden answer.
        options: Multiple-choice options, e.g. ["A) ...", "B) ..."].
        answer: Gold answer, usually "A"/"B"/"C"/"D" for public toy data.
        subject: Topic label, e.g. geometry/algebra/plots.
        source: Dataset/source label.
    """

    id: str
    image: Image.Image
    question: str
    options: list[str]
    answer: str
    subject: str
    source: str = "unknown"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a jsonl file.

    This helper is provided; you may use or replace it.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc
    return rows


def sanitize_question(text: str) -> str:
    """Remove image/control tokens that must not appear in raw questions."""
    for token in ("<image>", "<image_start>", "<image_end>"):
        text = text.replace(token, "")
    return " ".join(text.split())


class MathVQADataset(Dataset[MathVQASample]):
    """Dataset for manifest-based visual mathematical QA.

    Expected manifest fields:
        id, split, image, question, options, answer, subject, source(optional)

    TODO for students:
        - read manifest;
        - filter by split;
        - support max_samples;
        - resolve image paths relative to manifest directory;
        - open images as RGB PIL.Image;
        - return MathVQASample.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.split = split
        self.max_samples = max_samples

        rows = [r for r in load_jsonl(self.manifest_path) if r.get("split") == split]
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> MathVQASample:
        row = self.rows[idx]
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = Image.open(image_path).convert("RGB")
        return MathVQASample(
            id=str(row.get("id", idx)),
            image=image,
            question=sanitize_question(str(row.get("question", ""))),
            options=list(row.get("options") or []),
            answer=str(row.get("answer", "")),
            subject=str(row.get("subject", "unknown")),
            source=str(row.get("source", "unknown")),
        )
