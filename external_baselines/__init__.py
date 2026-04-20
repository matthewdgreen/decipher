"""Sidecar harness for external cipher-solving baselines.

This package is intentionally outside ``src/`` so external solver experiments
do not become part of Decipher's core runtime.
"""

from external_baselines.harness import (
    ExternalBaselineConfig,
    ExternalBaselineResult,
    PreparedExternalCase,
    run_external_baseline,
)

__all__ = [
    "ExternalBaselineConfig",
    "ExternalBaselineResult",
    "PreparedExternalCase",
    "run_external_baseline",
]
