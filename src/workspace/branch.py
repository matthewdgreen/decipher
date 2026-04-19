"""Branch: a named partial key within a Workspace."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Branch:
    """A named partial key, forkable from another branch."""

    name: str
    key: dict[int, int] = field(default_factory=dict)  # ct_id -> pt_id
    parent: str | None = None
    created_iteration: int = 0
    tags: list[str] = field(default_factory=list)

    def copy_as(self, new_name: str, iteration: int) -> "Branch":
        return Branch(
            name=new_name,
            key=dict(self.key),
            parent=self.name,
            created_iteration=iteration,
            tags=list(self.tags),
        )

    def snapshot_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parent": self.parent,
            "mapped_count": len(self.key),
            "created_iteration": self.created_iteration,
            "tags": list(self.tags),
        }
