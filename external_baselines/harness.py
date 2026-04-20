"""Run external cipher solvers against generated Decipher test cases.

The harness is deliberately command-template based. Tools such as Zenith and
zkdecrypto-lite have their own release layouts and CLI conventions, so this
layer prepares common input files and lets the caller provide the exact command
for the locally installed binary or jar.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.loader import TestData
from benchmark.scorer import has_word_boundaries, score_decryption


_TEXT_EXTENSIONS = {".txt", ".out", ".log", ".json", ".csv"}


@dataclass
class ExternalBaselineConfig:
    """Configuration for one external solver invocation."""

    name: str
    command: list[str]
    timeout_seconds: int = 300
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    solution_regex: str | None = None
    selection: str = "first"  # first | last | best_by_score

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExternalBaselineConfig":
        command = data.get("command")
        if isinstance(command, str):
            command = shlex.split(command)
        if not command:
            raise ValueError(f"External baseline {data.get('name', '<unnamed>')} has no command")
        return cls(
            name=str(data["name"]),
            command=[str(part) for part in command],
            timeout_seconds=int(data.get("timeout_seconds", 300)),
            cwd=data.get("cwd"),
            env={str(k): str(v) for k, v in data.get("env", {}).items()},
            solution_regex=data.get("solution_regex"),
            selection=str(data.get("selection", "first")),
        )


@dataclass
class PreparedExternalCase:
    """Files made available to an external solver."""

    test_id: str
    work_dir: Path
    input_dir: Path
    output_dir: Path
    canonical_file: Path
    tokens_file: Path
    compact_file: Path
    letters_file: Path
    metadata_file: Path
    ground_truth_file: Path
    output_file: Path


@dataclass
class ExternalBaselineResult:
    test_id: str
    solver: str
    status: str
    char_accuracy: float
    word_accuracy: float
    elapsed: float
    decryption: str
    ground_truth: str
    command: list[str]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    artifact_path: str = ""
    selection: str = "first"
    oracle_selection: bool = False
    candidates_considered: int = 0

    def save(self, path: str | Path, prepared: PreparedExternalCase | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "test_id": self.test_id,
            "solver": self.solver,
            "status": self.status,
            "char_accuracy": self.char_accuracy,
            "word_accuracy": self.word_accuracy,
            "elapsed": round(self.elapsed, 3),
            "decryption": self.decryption,
            "ground_truth": self.ground_truth,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "selection": self.selection,
            "oracle_selection": self.oracle_selection,
            "candidates_considered": self.candidates_considered,
        }
        if prepared is not None:
            data["prepared_case"] = {
                "work_dir": str(prepared.work_dir),
                "input_dir": str(prepared.input_dir),
                "output_dir": str(prepared.output_dir),
                "canonical_file": str(prepared.canonical_file),
                "tokens_file": str(prepared.tokens_file),
                "compact_file": str(prepared.compact_file),
                "letters_file": str(prepared.letters_file),
                "metadata_file": str(prepared.metadata_file),
                "ground_truth_file": str(prepared.ground_truth_file),
                "output_file": str(prepared.output_file),
            }
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_external_case(
    test_data: TestData,
    base_dir: str | Path,
    solver_name: str,
) -> PreparedExternalCase:
    """Create a self-contained directory of solver-neutral input files."""
    test_id = test_data.test.test_id
    run_id = uuid.uuid4().hex[:10]
    safe_solver = re.sub(r"[^A-Za-z0-9_.-]+", "_", solver_name).strip("_") or "solver"
    work_dir = Path(base_dir) / test_id / f"{safe_solver}_{run_id}"
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = _canonical_tokens(test_data.canonical_transcription)
    letters = _symbolize_tokens(tokens)
    metadata = {
        "test_id": test_id,
        "description": test_data.test.description,
        "cipher_system": test_data.test.cipher_system,
        "word_boundaries": has_word_boundaries(test_data.plaintext),
        "symbol_count": len(set(tokens)),
        "token_count": len(tokens),
        "format_notes": {
            "canonical.txt": "Original Decipher transcription, with | marking word boundaries when present.",
            "tokens.txt": "Whitespace-separated cipher symbols with word separators removed.",
            "compact.txt": "Cipher symbols concatenated without separators; useful only when symbols are fixed-width or single-character.",
            "letters.txt": "Same token stream remapped to single-character symbols for solvers that cannot read multi-character tokens.",
            "symbol_map": "Maps original cipher symbols to letters.txt symbols.",
        },
        "symbol_map": dict(zip(tokens, letters, strict=False)) if len(tokens) == len(letters) else {},
    }

    canonical_file = input_dir / "canonical.txt"
    tokens_file = input_dir / "tokens.txt"
    compact_file = input_dir / "compact.txt"
    letters_file = input_dir / "letters.txt"
    metadata_file = input_dir / "metadata.json"
    ground_truth_file = input_dir / "ground_truth.txt"
    output_file = output_dir / "solution.txt"

    canonical_file.write_text(test_data.canonical_transcription + "\n", encoding="utf-8")
    tokens_file.write_text(" ".join(tokens) + "\n", encoding="utf-8")
    compact_file.write_text("".join(tokens) + "\n", encoding="utf-8")
    letters_file.write_text("".join(letters) + "\n", encoding="utf-8")
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    ground_truth_file.write_text(test_data.plaintext + "\n", encoding="utf-8")

    return PreparedExternalCase(
        test_id=test_id,
        work_dir=work_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        canonical_file=canonical_file,
        tokens_file=tokens_file,
        compact_file=compact_file,
        letters_file=letters_file,
        metadata_file=metadata_file,
        ground_truth_file=ground_truth_file,
        output_file=output_file,
    )


def run_external_baseline(
    test_data: TestData,
    config: ExternalBaselineConfig,
    artifact_dir: str | Path = "artifacts/external_baselines",
) -> ExternalBaselineResult:
    """Run a configured external solver and score the extracted plaintext."""
    prepared = prepare_external_case(test_data, artifact_dir, config.name)
    command = [_expand_placeholders(part, test_data, prepared) for part in config.command]
    env = os.environ.copy()
    env.update({k: _expand_placeholders(v, test_data, prepared) for k, v in config.env.items()})
    cwd = _expand_placeholders(config.cwd, test_data, prepared) if config.cwd else None

    t0 = time.time()
    stdout = ""
    stderr = ""
    error = ""
    returncode: int | None = None
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        error = f"Timed out after {config.timeout_seconds}s"
    except FileNotFoundError as exc:
        error = f"Command not found: {exc.filename}"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    elapsed = time.time() - t0

    captured_text = _collect_solver_text(stdout, stderr, prepared)
    candidates = extract_candidates(captured_text, config.solution_regex)
    decryption, oracle = _select_candidate(
        candidates,
        test_data.plaintext,
        config.selection,
        test_data.test.test_id,
    )
    status = "error" if error else ("failed" if returncode not in (0, None) else "completed")
    if decryption:
        status = "solved" if _char_accuracy(test_data.test.test_id, decryption, test_data.plaintext) >= 1.0 else status

    score = score_decryption(
        test_id=test_data.test.test_id,
        decrypted=decryption,
        ground_truth=test_data.plaintext,
        agent_score=0.0,
        status=status,
    )
    result = ExternalBaselineResult(
        test_id=test_data.test.test_id,
        solver=config.name,
        status=status,
        char_accuracy=score.char_accuracy,
        word_accuracy=score.word_accuracy,
        elapsed=elapsed,
        decryption=decryption,
        ground_truth=test_data.plaintext,
        command=command,
        returncode=returncode,
        stdout=_truncate(stdout, 12000),
        stderr=_truncate(stderr, 12000),
        error=error,
        selection=config.selection,
        oracle_selection=oracle,
        candidates_considered=len(candidates),
    )

    artifact_path = prepared.work_dir / "artifact.json"
    result.save(artifact_path, prepared)
    result.artifact_path = str(artifact_path)
    return result


def extract_candidates(text: str, solution_regex: str | None = None) -> list[str]:
    """Extract plausible plaintext candidates from solver output text."""
    candidates: list[str] = []
    if solution_regex:
        regex = re.compile(solution_regex, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        for match in regex.finditer(text):
            if "plaintext" in match.groupdict():
                candidates.append(match.group("plaintext"))
            elif match.groups():
                candidates.append(match.group(1))
            else:
                candidates.append(match.group(0))

    candidates.extend(_json_candidates(text))

    label_re = re.compile(
        r"(?im)^\s*(?:solution|plaintext|plain\s*text|decryption|best)\s*[:=]\s*(.+?)\s*$"
    )
    candidates.extend(match.group(1) for match in label_re.finditer(text))

    if not candidates:
        candidates.extend(_line_candidates(text))

    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        c = _clean_plaintext(candidate)
        if len(c.replace(" ", "")) < 10:
            continue
        if c not in seen:
            cleaned.append(c)
            seen.add(c)
    return cleaned


def _canonical_tokens(canonical: str) -> list[str]:
    return [tok for tok in canonical.replace("|", " ").split() if tok]


def _symbolize_tokens(tokens: list[str]) -> list[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    mapping: dict[str, str] = {}
    out: list[str] = []
    for token in tokens:
        if token not in mapping:
            if len(mapping) >= len(alphabet):
                raise ValueError("Too many unique symbols to fit letters.txt single-character alphabet")
            mapping[token] = alphabet[len(mapping)]
        out.append(mapping[token])
    return out


def _expand_placeholders(
    value: str | None,
    test_data: TestData,
    prepared: PreparedExternalCase,
) -> str:
    if value is None:
        return ""
    replacements = {
        "test_id": test_data.test.test_id,
        "cipher_system": test_data.test.cipher_system,
        "canonical_file": str(prepared.canonical_file),
        "tokens_file": str(prepared.tokens_file),
        "compact_file": str(prepared.compact_file),
        "letters_file": str(prepared.letters_file),
        "metadata_file": str(prepared.metadata_file),
        "ground_truth_file": str(prepared.ground_truth_file),
        "input_dir": str(prepared.input_dir),
        "output_dir": str(prepared.output_dir),
        "output_file": str(prepared.output_file),
        "work_dir": str(prepared.work_dir),
    }
    return value.format(**replacements)


def _collect_solver_text(stdout: str, stderr: str, prepared: PreparedExternalCase) -> str:
    parts = [stdout, stderr]
    if prepared.output_file.exists():
        parts.append(prepared.output_file.read_text(encoding="utf-8", errors="replace"))
    for path in sorted(prepared.output_dir.rglob("*")):
        if not path.is_file() or path == prepared.output_file:
            continue
        if path.suffix.lower() in _TEXT_EXTENSIONS and path.stat().st_size <= 2_000_000:
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(part for part in parts if part)


def _json_candidates(text: str) -> list[str]:
    out: list[str] = []
    keys = {"plaintext", "plainText", "decryption", "solution", "bestPlaintext", "decoded", "text"}
    for obj_text in _probable_json_objects(text):
        try:
            data = json.loads(obj_text)
        except json.JSONDecodeError:
            continue
        out.extend(_walk_json_for_strings(data, keys))
    return out


def _probable_json_objects(text: str) -> list[str]:
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        return [stripped]
    return re.findall(r"(?s)(\{.*?\}|\[.*?\])", text)


def _walk_json_for_strings(data: Any, keys: set[str]) -> list[str]:
    out: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys and isinstance(value, str):
                out.append(value)
            else:
                out.extend(_walk_json_for_strings(value, keys))
    elif isinstance(data, list):
        for item in data:
            out.extend(_walk_json_for_strings(item, keys))
    return out


def _line_candidates(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        cleaned = _clean_plaintext(line)
        letters = cleaned.replace(" ", "")
        if len(letters) >= 30 and len(letters) / max(1, len(line.strip())) > 0.7:
            out.append(cleaned)
    return out


def _clean_plaintext(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z ]", "", text).upper()
    return " ".join(cleaned.split())


def _select_candidate(
    candidates: list[str],
    ground_truth: str,
    selection: str,
    test_id: str,
) -> tuple[str, bool]:
    if not candidates:
        return "", False
    if selection == "last":
        return candidates[-1], False
    if selection == "best_by_score":
        best = max(candidates, key=lambda c: _char_accuracy(test_id, c, ground_truth))
        return best, True
    return candidates[0], False


def _char_accuracy(test_id: str, decrypted: str, ground_truth: str) -> float:
    score = score_decryption(
        test_id=test_id,
        decrypted=decrypted,
        ground_truth=ground_truth,
        agent_score=0.0,
        status="",
    )
    return score.char_accuracy


def _truncate(text: str, limit: int) -> str:
    return text[:limit] + ("\n...[truncated]" if len(text) > limit else "")
