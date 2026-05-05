"""Backward-compatible import path for Copiale null-mask experiments.

The shared implementation now lives in :mod:`analysis.homophonic_nulls`.
This shim keeps older scripts, notebooks, and artifacts importable while the
public refinement name moves to the more general ``null_masks`` spelling.
"""
from __future__ import annotations

from analysis.homophonic_nulls import *  # noqa: F401,F403

