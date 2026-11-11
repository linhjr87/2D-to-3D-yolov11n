"""Versioned class table shared by every pipeline stage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

VALID_GROUPS = frozenset(
    {
        "door",
        "window",
        "furniture",
        "sanitary-kitchen",
        "appliance-hvac",
        "vertical-circulation",
        "wall",
    }
)
VALID_MODELS = ("D", "W")


@dataclass(frozen=True)
class ClassDef:
    schema_id: int
    semantic_id: int
    name: str
    group: str
    model: Literal["D", "W"]
    enabled: bool = True


@dataclass(frozen=True)
class Schema:
    version: str
    classes: tuple[ClassDef, ...]

    def for_model(self, model: str) -> list[ClassDef]:
        """Enabled classes of a model ordered by schema_id; YOLO index = list position."""
        return sorted(
            (c for c in self.classes if c.model == model and c.enabled),
            key=lambda c: c.schema_id,
        )

    def yolo_index(self, model: str) -> dict[int, int]:
        return {c.semantic_id: i for i, c in enumerate(self.for_model(model))}

    def yolo_names(self, model: str) -> dict[int, str]:
        return {i: c.name for i, c in enumerate(self.for_model(model))}

    def by_schema_id(self, sid: int) -> ClassDef:
        for c in self.classes:
            if c.schema_id == sid:
                return c
        raise KeyError(f"unknown schema_id {sid}")


def _validate(classes: tuple[ClassDef, ...]) -> None:
    seen_sid: set[int] = set()
    seen_sem: set[tuple[str, int]] = set()
    for c in classes:
        if c.group not in VALID_GROUPS:
            raise ValueError(f"class {c.name!r}: invalid group {c.group!r}")
        if c.model not in VALID_MODELS:
            raise ValueError(f"class {c.name!r}: invalid model {c.model!r}")
        if c.schema_id in seen_sid:
            raise ValueError(f"duplicate schema_id {c.schema_id}")
        seen_sid.add(c.schema_id)
        key = (c.model, c.semantic_id)
        if key in seen_sem:
            raise ValueError(f"duplicate (model, semantic_id) {key}")
        seen_sem.add(key)


def load_schema(path: Path) -> Schema:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    classes = tuple(
        ClassDef(
            schema_id=int(r["schema_id"]),
            semantic_id=int(r["semantic_id"]),
            name=str(r["name"]),
            group=str(r["group"]),
            model=r["model"],
            enabled=bool(r.get("enabled", True)),
        )
        for r in doc["classes"]
    )
    _validate(classes)
    return Schema(version=str(doc["version"]), classes=classes)
