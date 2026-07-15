from __future__ import annotations

import argparse
import json
import os

import pytest

from cli import HOMOPHONIC_REFINEMENT_CHOICES, cmd_benchmark


def test_cli_accepts_null_mask_homophonic_refinement_aliases():
    assert "null_masks" in HOMOPHONIC_REFINEMENT_CHOICES
    assert "homophonic_nulls" in HOMOPHONIC_REFINEMENT_CHOICES
    assert "copiale_nulls" in HOMOPHONIC_REFINEMENT_CHOICES


def _minimal_benchmark_dir(tmp_path):
    """A benchmark root with one manifest record and one split test.

    The split's single test id is ``real_page1``; anything else the caller
    filters for will not match, exercising the "No matching tests" path.
    """
    os.makedirs(tmp_path / "manifest")
    os.makedirs(tmp_path / "splits")
    record = {
        "id": "page1",
        "source": "real",
        "cipher_type": ["monoalphabetic_substitution"],
        "plaintext_language": "en",
        "transcription_canonical_file": "",
        "plaintext_file": "",
        "has_key": True,
    }
    (tmp_path / "manifest" / "records.jsonl").write_text(json.dumps(record) + "\n")
    test_def = {
        "test_id": "real_page1",
        "track": "transcription2plaintext",
        "cipher_system": "test_cipher",
        "target_records": ["page1"],
        "context_records": [],
        "description": "real page",
    }
    (tmp_path / "splits" / "all_tests.jsonl").write_text(json.dumps(test_def) + "\n")
    return tmp_path


def _benchmark_args(benchmark_path, **overrides):
    defaults = dict(
        agentic=False,
        multipage_group=None,
        analyze=False,
        benchmark_path=str(benchmark_path),
        split=None,
        source=None,
        track="transcription2plaintext",
        test_id=None,
        limit=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_benchmark_no_match_message_names_split_and_filters_and_synth_hint(tmp_path, capsys):
    """A synthetic --test-id with no --split fails with a diagnostic message.

    The message must name the auto-detected split file that was searched, the
    --test-id filter applied, and hint that synthetic ids need an explicit
    --split.
    """
    bench = _minimal_benchmark_dir(tmp_path)
    args = _benchmark_args(bench, test_id="synth_en_200honb_s6")

    with pytest.raises(SystemExit) as excinfo:
        cmd_benchmark(args)
    assert excinfo.value.code == 1

    err = capsys.readouterr().err
    # Names the split that was auto-detected and searched.
    assert "all_tests.jsonl" in err
    assert "default (no --source or --split given)" in err
    assert str((bench / "splits" / "all_tests.jsonl")) in err
    # Names the filter that was applied.
    assert "--test-id synth_en_200honb_s6" in err
    # Hints that synthetic ids need an explicit --split.
    assert "--split en_ss_synth_nb_tests.jsonl" in err


def test_benchmark_no_match_message_omits_synth_hint_when_split_explicit(tmp_path, capsys):
    """When --split is explicit, no synthetic-split hint is shown."""
    bench = _minimal_benchmark_dir(tmp_path)
    args = _benchmark_args(bench, split="all_tests.jsonl", test_id="nope")

    with pytest.raises(SystemExit) as excinfo:
        cmd_benchmark(args)
    assert excinfo.value.code == 1

    err = capsys.readouterr().err
    assert "specified via --split" in err
    assert "--test-id nope" in err
    assert "Hint: synthetic test ids" not in err


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
