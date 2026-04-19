"""Workspace: the stateful environment a v2 agent inhabits."""
from __future__ import annotations

from typing import Any

from models.alphabet import Alphabet
from models.cipher_text import CipherText
from workspace.branch import Branch


class WorkspaceError(Exception):
    pass


class Workspace:
    """Cipher + named branches. No Qt dependency; pure data.

    Branches are independent partial keys. The "main" branch is created
    automatically at construction. All mutation and decoding methods take
    an explicit branch name — there is no implicit current branch.
    """

    def __init__(
        self,
        cipher_text: CipherText,
        plaintext_alphabet: Alphabet | None = None,
    ) -> None:
        self.cipher_text = cipher_text
        self.plaintext_alphabet = plaintext_alphabet or Alphabet.standard_english()
        self._branches: dict[str, Branch] = {
            "main": Branch(name="main", key={}, parent=None, created_iteration=0)
        }
        self._iteration: int = 0

    # --- iteration tracking (for branch provenance) ---

    def set_iteration(self, n: int) -> None:
        self._iteration = n

    # --- branch inspection ---

    def has_branch(self, name: str) -> bool:
        return name in self._branches

    def get_branch(self, name: str) -> Branch:
        if name not in self._branches:
            raise WorkspaceError(f"Branch not found: {name}")
        return self._branches[name]

    def list_branches(self) -> list[dict[str, Any]]:
        return [b.snapshot_dict() for b in self._branches.values()]

    def branch_names(self) -> list[str]:
        return list(self._branches.keys())

    # --- branch lifecycle ---

    def fork(self, new_name: str, from_branch: str = "main") -> Branch:
        if new_name in self._branches:
            raise WorkspaceError(f"Branch already exists: {new_name}")
        if not new_name or not new_name.replace("_", "").replace("-", "").isalnum():
            raise WorkspaceError(
                f"Branch name must be alphanumeric (plus _ or -): {new_name!r}"
            )
        src = self.get_branch(from_branch)
        new_branch = src.copy_as(new_name, self._iteration)
        self._branches[new_name] = new_branch
        return new_branch

    def delete(self, name: str) -> None:
        if name == "main":
            raise WorkspaceError("Cannot delete 'main' branch")
        if name not in self._branches:
            raise WorkspaceError(f"Branch not found: {name}")
        del self._branches[name]

    def tag(self, name: str, tag: str) -> None:
        branch = self.get_branch(name)
        if tag not in branch.tags:
            branch.tags.append(tag)

    # --- key mutation (always scoped to a branch) ---

    def set_mapping(self, branch_name: str, ct_id: int, pt_id: int) -> None:
        branch = self.get_branch(branch_name)
        branch.key[ct_id] = pt_id

    def clear_mapping(self, branch_name: str, ct_id: int) -> None:
        branch = self.get_branch(branch_name)
        branch.key.pop(ct_id, None)

    def set_full_key(self, branch_name: str, key: dict[int, int]) -> None:
        branch = self.get_branch(branch_name)
        branch.key = dict(key)

    # --- decoding ---

    def apply_key(self, branch_name: str) -> str:
        branch = self.get_branch(branch_name)
        return self._decode_with(branch.key)

    def _decode_with(self, key: dict[int, int]) -> str:
        ct = self.cipher_text
        pt_alpha = self.plaintext_alphabet
        sep_inner = " " if pt_alpha._multisym else ""

        def decode_word(word_tokens: list[int]) -> str:
            parts: list[str] = []
            for t in word_tokens:
                if t in key:
                    parts.append(pt_alpha.symbol_for(key[t]))
                else:
                    parts.append("?")
            return sep_inner.join(parts)

        word_sep = ct.separator or ""
        return word_sep.join(decode_word(w) for w in ct.words)

    def is_complete(self, branch_name: str) -> bool:
        branch = self.get_branch(branch_name)
        used = set(self.cipher_text.tokens)
        return bool(used) and all(s in branch.key for s in used)

    def unmapped_cipher_ids(self, branch_name: str) -> list[int]:
        branch = self.get_branch(branch_name)
        used = set(self.cipher_text.tokens)
        return sorted(used - set(branch.key.keys()))

    def unused_plain_ids(self, branch_name: str) -> list[int]:
        branch = self.get_branch(branch_name)
        used = set(branch.key.values())
        return [i for i in range(self.plaintext_alphabet.size) if i not in used]

    # --- compare + merge ---

    def compare(self, branch_a: str, branch_b: str) -> dict[str, Any]:
        ba = self.get_branch(branch_a)
        bb = self.get_branch(branch_b)

        a_keys = set(ba.key.keys())
        b_keys = set(bb.key.keys())
        shared_keys = a_keys & b_keys

        agreements = sum(1 for k in shared_keys if ba.key[k] == bb.key[k])
        disagreements = len(shared_keys) - agreements
        a_only = len(a_keys - b_keys)
        b_only = len(b_keys - a_keys)

        return {
            "branch_a": branch_a,
            "branch_b": branch_b,
            "a_mapped_count": len(ba.key),
            "b_mapped_count": len(bb.key),
            "agreements": agreements,
            "disagreements": disagreements,
            "a_only": a_only,
            "b_only": b_only,
            "a_decryption": self.apply_key(branch_a),
            "b_decryption": self.apply_key(branch_b),
        }

    def merge(
        self,
        from_branch: str,
        into_branch: str,
        policy: str = "non_conflicting",
    ) -> dict[str, Any]:
        """Copy mappings from `from_branch` into `into_branch`.

        Policies:
          - non_conflicting: only copy mappings whose ct_id is not yet mapped
                             in dst (ct_id conflict skipped; pt_id duplicates
                             are allowed — homophonic ciphers are supported)
          - override:        src wins on all conflicts
          - keep:            dst wins on all conflicts (only adds new ct_ids)
        """
        if policy not in {"non_conflicting", "override", "keep"}:
            raise WorkspaceError(f"Unknown merge policy: {policy}")
        src = self.get_branch(from_branch)
        dst = self.get_branch(into_branch)

        added = 0
        overwritten = 0
        skipped = 0

        for ct_id, pt_id in src.key.items():
            if ct_id in dst.key:
                if dst.key[ct_id] == pt_id:
                    continue
                if policy == "override":
                    dst.key[ct_id] = pt_id
                    overwritten += 1
                else:
                    skipped += 1
            else:
                dst.key[ct_id] = pt_id
                added += 1

        return {
            "from_branch": from_branch,
            "into_branch": into_branch,
            "policy": policy,
            "added": added,
            "overwritten": overwritten,
            "skipped": skipped,
            "into_mapped_count": len(dst.key),
        }

    # --- summary for artifacts ---

    def snapshot_branch(self, name: str) -> dict[str, Any]:
        branch = self.get_branch(name)
        return {
            "name": name,
            "parent": branch.parent,
            "created_iteration": branch.created_iteration,
            "key": dict(branch.key),
            "mapped_count": len(branch.key),
            "decryption": self.apply_key(name),
            "tags": list(branch.tags),
        }
