"""Family + discriminator registry invariants (INV-0 Part 1 / Part 9)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from investigation import families as F


def test_registry_imports_and_validates():
    # _validate_registry() runs at import; reaching here means it passed.
    assert len(F.PRIMARY_IDS) == 12
    assert len(F.SUBTYPE_IDS) == 4
    assert len(F.MODIFIER_IDS) == 1


def test_every_primary_has_a_discriminator():
    for fid in F.PRIMARY_IDS:
        assert F.FAMILY_REGISTRY[fid].discriminators, fid


def test_confusables_symmetric():
    for fam in F.FAMILY_REGISTRY.values():
        for other in fam.confusable_with:
            assert fam.id in F.FAMILY_REGISTRY[other].confusable_with


def test_discriminator_splits_reference_real_families():
    for disc in F.DISCRIMINATOR_REGISTRY.values():
        for fid in disc.splits:
            assert fid in F.FAMILY_REGISTRY


def test_suspicion_unknown_maps_to_nothing():
    # The six real keys map; "unknown" is absent (dropped, emits no prior atom).
    assert "unknown" not in F.SUSPICION_TO_FAMILY
    assert F.SUSPICION_TO_FAMILY["polyalphabetic_vigenere"] == "polyalphabetic_periodic"
    assert len(F.SUSPICION_TO_FAMILY) == 6


def test_poly_discriminator_order_preserved():
    # fixture (ii)'s fallback picks disc_sub_periodic first -> it must lead.
    assert F.family_discriminators("polyalphabetic_periodic")[0] == "disc_sub_periodic"


def test_available_discriminators_present():
    for did in ("disc_mono_transp", "disc_mono_homophonic", "disc_sub_periodic",
                "disc_periodic_quagmire", "disc_homo_transphomo",
                "disc_numeric_book_hoax", "disc_book_word_char"):
        assert F.DISCRIMINATOR_REGISTRY[did].status == "available"


def test_substitution_primaries_are_eight():
    assert len(F.SUBSTITUTION_PRIMARIES) == 8
    assert "numeric_book_cipher" not in F.SUBSTITUTION_PRIMARIES
