"""Branch: a named partial key within a Workspace."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Branch:
    """A named partial key, forkable from another branch."""

    name: str
    key: dict[int, int] = field(default_factory=dict)  # ct_id -> pt_id
    word_spans: list[tuple[int, int]] | None = None
    token_order: list[int] | None = None
    transform_pipeline: dict | None = None
    parent: str | None = None
    created_iteration: int = 0
    tags: list[str] = field(default_factory=list)

    def copy_as(self, new_name: str, iteration: int) -> "Branch":
        return Branch(
            name=new_name,
            key=dict(self.key),
            word_spans=list(self.word_spans) if self.word_spans is not None else None,
            token_order=list(self.token_order) if self.token_order is not None else None,
            transform_pipeline=dict(self.transform_pipeline) if self.transform_pipeline is not None else None,
            parent=self.name,
            created_iteration=iteration,
            tags=list(self.tags),
        )

    def snapshot_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parent": self.parent,
            "mapped_count": len(self.key),
            "custom_word_boundaries": self.word_spans is not None,
            "custom_token_order": self.token_order is not None,
            "transform_pipeline": self.transform_pipeline,
            "word_count": len(self.word_spans) if self.word_spans is not None else None,
            "created_iteration": self.created_iteration,
            "tags": list(self.tags),
        }
