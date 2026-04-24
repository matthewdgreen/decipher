from __future__ import annotations

import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def verify_model(path: Path) -> dict[str, object]:
    _ensure_src_on_path()
    from analysis.zenith_solver import load_zenith_binary_model

    model = load_zenith_binary_model(path)
    letter_total = sum(model.letter_freq.values())
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks = {
        "shape_ok": model.log_probs.shape == (26 ** 5,),
        "letter_freq_total": letter_total,
        "unknown_log_prob": model.unknown_log_prob,
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
        "metadata": metadata,
        "sample_ation": model.lookup("ation"),
        "sample_there": model.lookup("there"),
    }
    if not checks["shape_ok"]:
        raise ValueError("model array shape mismatch")
    if not (0.99 < letter_total < 1.01):
        raise ValueError(f"letter frequencies sum to {letter_total}, expected ~1.0")
    return checks
