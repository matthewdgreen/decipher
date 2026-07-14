from __future__ import annotations

from cli import HOMOPHONIC_REFINEMENT_CHOICES


def test_cli_accepts_null_mask_homophonic_refinement_aliases():
    assert "null_masks" in HOMOPHONIC_REFINEMENT_CHOICES
    assert "homophonic_nulls" in HOMOPHONIC_REFINEMENT_CHOICES
    assert "copiale_nulls" in HOMOPHONIC_REFINEMENT_CHOICES


def test_diagnose_numeric_stdin_json():
    """`decipher diagnose - --json` on a numeric stream ranks numeric_book top."""
    import json
    import os
    import subprocess
    import sys

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = dict(os.environ, PYTHONPATH=os.path.join(repo, "src"))
    # Early-biased (front-loaded) numeric book-cipher-like stream.
    import random
    rng = random.Random(7)
    vals = []
    for _ in range(300):
        vals.append(rng.randint(1, 260) if rng.random() < 0.55 else rng.randint(1, 1320))
    stdin = " ".join(str(v) for v in vals)

    out = subprocess.run(
        [sys.executable, "-m", "cli", "diagnose", "-", "--json"],
        input=stdin, capture_output=True, text=True, cwd=repo, env=env,
    )
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert report["alphabet_class"] == "numeric"
    assert report["ranked"][0]["family"] == "numeric_book_cipher"
    assert report["battery_coverage"]["numeric_code"] == "ran"
