from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_automated_parity_matrix.py"
spec = importlib.util.spec_from_file_location("run_automated_parity_matrix", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
matrix = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = matrix
spec.loader.exec_module(matrix)


def test_parse_int_ranges_supports_lists_and_ranges():
    assert matrix.parse_int_ranges("1-3,7,10-8") == [1, 2, 3, 7, 10, 9, 8]


def test_select_chunk_applies_shard_then_offset_limit():
    cases = list(range(10))

    selected = matrix.select_chunk(cases, offset=1, limit=2, shard_index=1, shard_count=3)

    assert selected == [4, 7]
