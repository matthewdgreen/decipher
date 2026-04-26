"""Workspace: the stateful environment a v2 agent inhabits."""
from __future__ import annotations

from typing import Any

from analysis.transformers import TransformPipeline, apply_transform_pipeline
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

    def _default_word_spans(self, branch_name: str | None = None) -> list[tuple[int, int]]:
        if branch_name is not None:
            branch = self.get_branch(branch_name)
            if branch.token_order is not None:
                return [(0, len(branch.token_order))] if branch.token_order else []
        spans: list[tuple[int, int]] = []
        start = 0
        for word in self.cipher_text.words:
            end = start + len(word)
            spans.append((start, end))
            start = end
        return spans

    def _base_flat_word_tokens(self) -> list[int]:
        flat: list[int] = []
        for word in self.cipher_text.words:
            flat.extend(word)
        return flat

    def effective_tokens(self, branch_name: str = "main") -> list[int]:
        branch = self.get_branch(branch_name)
        base_tokens = self._base_flat_word_tokens()
        if branch.token_order is None:
            return base_tokens
        return [base_tokens[i] for i in branch.token_order]

    def effective_cipher_text(self, branch_name: str = "main") -> CipherText:
        branch = self.get_branch(branch_name)
        if branch.token_order is None and branch.word_spans is None:
            return self.cipher_text
        tokens = self.effective_tokens(branch_name)
        raw = self.cipher_text.alphabet.decode(tokens)
        return CipherText(
            raw=raw,
            alphabet=self.cipher_text.alphabet,
            source=f"{self.cipher_text.source}:branch:{branch_name}",
            separator=None,
        )

    def _flat_word_tokens(self, branch_name: str | None = None) -> list[int]:
        if branch_name is not None:
            return self.effective_tokens(branch_name)
        return self._base_flat_word_tokens()

    def _normalize_word_spans(
        self,
        spans: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        tokens = self._flat_word_tokens(branch_name=None)
        if not tokens:
            if spans:
                raise WorkspaceError("Word spans must be empty for empty ciphertext")
            return []
        if not spans:
            raise WorkspaceError("Word spans may not be empty for non-empty ciphertext")

        normalized: list[tuple[int, int]] = []
        expected_start = 0
        total = len(tokens)
        for start, end in spans:
            start = int(start)
            end = int(end)
            if start != expected_start:
                raise WorkspaceError("Word spans must be contiguous and gap-free")
            if end <= start:
                raise WorkspaceError("Word spans must have positive length")
            if end > total:
                raise WorkspaceError("Word spans exceed ciphertext length")
            normalized.append((start, end))
            expected_start = end
        if expected_start != total:
            raise WorkspaceError("Word spans must cover all ciphertext tokens exactly once")
        return normalized

    def effective_word_spans(self, branch_name: str) -> list[tuple[int, int]]:
        branch = self.get_branch(branch_name)
        if branch.word_spans is not None:
            return list(branch.word_spans)
        return self._default_word_spans(branch_name)

    def effective_words(self, branch_name: str) -> list[list[int]]:
        spans = self.effective_word_spans(branch_name)
        tokens = self._flat_word_tokens(branch_name)
        return [tokens[start:end] for start, end in spans]

    def set_word_spans(self, branch_name: str, spans: list[tuple[int, int]]) -> None:
        branch = self.get_branch(branch_name)
        normalized = self._normalize_word_spans(spans)
        default = self._default_word_spans(branch_name)
        branch.word_spans = None if normalized == default else normalized

    def reset_word_spans(self, branch_name: str) -> None:
        branch = self.get_branch(branch_name)
        branch.word_spans = None

    def apply_transform_pipeline(
        self,
        branch_name: str,
        pipeline: TransformPipeline | dict[str, Any] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        branch = self.get_branch(branch_name)
        parsed = TransformPipeline.from_raw(pipeline)
        if parsed is None or parsed.is_empty():
            branch.token_order = None
            branch.transform_pipeline = None
            branch.word_spans = None
            return {
                "branch": branch_name,
                "transformed": False,
                "token_count": len(self.cipher_text.tokens),
            }
        base_tokens = self._base_flat_word_tokens()
        order = apply_transform_pipeline(list(range(len(base_tokens))), parsed).tokens
        if sorted(order) != list(range(len(base_tokens))):
            raise WorkspaceError("Transform pipeline did not produce a token permutation")
        branch.token_order = order
        branch.transform_pipeline = parsed.to_raw()
        branch.word_spans = None
        return {
            "branch": branch_name,
            "transformed": True,
            "token_count": len(order),
            "pipeline": parsed.to_raw(),
        }

    def split_cipher_word(self, branch_name: str, word_index: int, split_at_offset: int) -> dict[str, Any]:
        spans = self.effective_word_spans(branch_name)
        if word_index < 0 or word_index >= len(spans):
            raise WorkspaceError(f"cipher_word_index {word_index} out of range")
        start, end = spans[word_index]
        word_len = end - start
        if split_at_offset <= 0 or split_at_offset >= word_len:
            raise WorkspaceError("split_at_offset must be inside the word")
        split_at = start + split_at_offset
        new_spans = spans[:word_index] + [(start, split_at), (split_at, end)] + spans[word_index + 1 :]
        self.set_word_spans(branch_name, new_spans)
        return {
            "word_index": word_index,
            "left_span": (start, split_at),
            "right_span": (split_at, end),
            "word_count": len(new_spans),
        }

    def merge_cipher_words(self, branch_name: str, left_word_index: int) -> dict[str, Any]:
        spans = self.effective_word_spans(branch_name)
        if left_word_index < 0 or left_word_index >= len(spans) - 1:
            raise WorkspaceError(f"left_word_index {left_word_index} out of range")
        left_start, left_end = spans[left_word_index]
        right_start, right_end = spans[left_word_index + 1]
        if left_end != right_start:
            raise WorkspaceError("Only adjacent words may be merged")
        new_spans = spans[:left_word_index] + [(left_start, right_end)] + spans[left_word_index + 2 :]
        self.set_word_spans(branch_name, new_spans)
        return {
            "left_word_index": left_word_index,
            "merged_span": (left_start, right_end),
            "word_count": len(new_spans),
        }

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

    def restore_branch(
        self,
        name: str,
        *,
        key: dict[int, int],
        parent: str | None = None,
        created_iteration: int = 0,
        tags: list[str] | None = None,
        word_spans: list[tuple[int, int]] | None = None,
        token_order: list[int] | None = None,
        transform_pipeline: dict | None = None,
    ) -> Branch:
        """Restore a branch from an artifact snapshot.

        This is intentionally a low-level state operation for continuation
        runs. It avoids replaying old tool calls while preserving the branch
        state the previous run actually ended with.
        """
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise WorkspaceError(
                f"Branch name must be alphanumeric (plus _ or -): {name!r}"
            )
        normalized_spans = (
            self._normalize_word_spans(word_spans) if word_spans is not None else None
        )
        branch = Branch(
            name=name,
            key=dict(key),
            word_spans=normalized_spans,
            token_order=list(token_order) if token_order is not None else None,
            transform_pipeline=dict(transform_pipeline) if transform_pipeline is not None else None,
            parent=parent,
            created_iteration=created_iteration,
            tags=list(tags or []),
        )
        self._branches[name] = branch
        return branch

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
        return self._decode_with(branch.key, self.effective_words(branch_name))

    def _decode_with(self, key: dict[int, int], words: list[list[int]] | None = None) -> str:
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
        use_words = words if words is not None else ct.words
        return word_sep.join(decode_word(w) for w in use_words)

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
            "word_spans": self.effective_word_spans(name),
            "custom_word_boundaries": branch.word_spans is not None,
            "mapped_count": len(branch.key),
            "decryption": self.apply_key(name),
            "tags": list(branch.tags),
        }
