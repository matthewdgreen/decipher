"""Transposition ciphers (ACA definitions).

Families: Columnar (complete + incomplete), Railfence/Redefence,
Route (spiral/diagonal), Amsco, Myszkowski, Cadenus, Nihilist transposition.

All operate on cleaned uppercase A-Z strings.  Columnar/Railfence/Redefence/
Route/Amsco/Myszkowski round-trip exactly on any length.  Cadenus and Nihilist
transposition use fixed-shape blocks and pad the final block with ``X``;
``decrypt(encrypt(pt))`` reproduces the *padded* text, exposed via ``prepare``.

References: ACA cipher guide ("Complete/Incomplete Columnar", "Railfence",
"Redefence", "Route", "Amsco", "Myszkowski", "Cadenus", "Nihilist
Transposition"); Kahn, *The Codebreakers*.

The module also exposes ``columnar_encrypt`` / ``columnar_decrypt`` reused by
the ADFGX/ADFGVX ciphers in ``ciphers.fractionation``.
"""

from __future__ import annotations

import math
import random
from typing import Any

from ciphers.family_base import FamilyCipher
from ciphers.textutil import clean, random_keyword


def _rank_order(key: str) -> list[int]:
    """Column read order: indices sorted by (key letter, position)."""
    return sorted(range(len(key)), key=lambda i: (key[i], i))


# ---------------------------------------------------------------------------
# Columnar transposition (complete + incomplete)
# ---------------------------------------------------------------------------

def columnar_encrypt(text: str, key: str) -> str:
    """Keyed columnar transposition (handles incomplete final row)."""
    ncols = len(key)
    if ncols == 0:
        raise ValueError("columnar key must be non-empty")
    cols = ["" for _ in range(ncols)]
    for i, ch in enumerate(text):
        cols[i % ncols] += ch
    return "".join(cols[j] for j in _rank_order(key))


def columnar_decrypt(text: str, key: str) -> str:
    ncols = len(key)
    if ncols == 0:
        raise ValueError("columnar key must be non-empty")
    n = len(text)
    base, extra = divmod(n, ncols)
    col_len = [base + (1 if j < extra else 0) for j in range(ncols)]
    cols = [""] * ncols
    idx = 0
    for j in _rank_order(key):
        cols[j] = text[idx: idx + col_len[j]]
        idx += col_len[j]
    out: list[str] = []
    maxlen = max(col_len) if col_len else 0
    for r in range(maxlen):
        for j in range(ncols):
            if r < len(cols[j]):
                out.append(cols[j][r])
    return "".join(out)


class ColumnarCipher(FamilyCipher):
    """Complete/incomplete keyed columnar transposition."""

    name = "columnar_transposition"

    def encrypt(self, plaintext: str, key: str) -> str:
        return columnar_encrypt(clean(plaintext), clean(key))

    def decrypt(self, ciphertext: str, key: str) -> str:
        return columnar_decrypt(clean(ciphertext), clean(key))

    def random_key(self, rng: random.Random, **_: Any) -> str:
        return random_keyword(rng, min_len=5, max_len=8)

    def describe_key(self, key: str) -> str:
        k = clean(key)
        return f"columnar keyword={k!r} order={_rank_order(k)}"

    def key_space_size(self, *, columns: int = 7, **_: Any) -> int:
        return math.factorial(columns)


# ---------------------------------------------------------------------------
# Railfence / Redefence
# ---------------------------------------------------------------------------

def _rail_pattern(n: int, rails: int) -> list[int]:
    pattern: list[int] = []
    r, d = 0, 1
    for _ in range(n):
        pattern.append(r)
        if r == 0:
            d = 1
        elif r == rails - 1:
            d = -1
        r += d
    return pattern


class RailfenceCipher(FamilyCipher):
    """Rail-fence (zigzag) transposition.  Key is the number of rails (>= 2)."""

    name = "railfence"

    def encrypt(self, plaintext: str, key: int) -> str:
        text = clean(plaintext)
        rails = int(key)
        if rails < 2:
            return text
        fence: list[list[str]] = [[] for _ in range(rails)]
        for ch, r in zip(text, _rail_pattern(len(text), rails)):
            fence[r].append(ch)
        return "".join("".join(row) for row in fence)

    def decrypt(self, ciphertext: str, key: int) -> str:
        text = clean(ciphertext)
        rails = int(key)
        if rails < 2:
            return text
        pattern = _rail_pattern(len(text), rails)
        counts = [pattern.count(k) for k in range(rails)]
        pos = 0
        rail_str: list[str] = []
        for k in range(rails):
            rail_str.append(text[pos: pos + counts[k]])
            pos += counts[k]
        idx = [0] * rails
        out: list[str] = []
        for r in pattern:
            out.append(rail_str[r][idx[r]])
            idx[r] += 1
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any) -> int:
        return rng.randint(3, 6)

    def describe_key(self, key: int) -> str:
        return f"railfence rails={int(key)}"

    def key_space_size(self, *, max_rails: int = 10, **_: Any) -> int:
        return max_rails - 1


class RedefenceCipher(FamilyCipher):
    """Redefence: rail-fence whose rails are read out in a keyed order.

    Key is ``(rails, order)`` where ``order`` is a permutation of
    ``range(rails)`` giving the rail read-out sequence.  (Rail-fence is the
    special case ``order == (0, 1, ..., rails-1)``.)
    """

    name = "redefence"

    def encrypt(self, plaintext: str, key) -> str:
        text = clean(plaintext)
        rails, order = int(key[0]), list(key[1])
        fence: list[list[str]] = [[] for _ in range(rails)]
        for ch, r in zip(text, _rail_pattern(len(text), rails)):
            fence[r].append(ch)
        return "".join("".join(fence[k]) for k in order)

    def decrypt(self, ciphertext: str, key) -> str:
        text = clean(ciphertext)
        rails, order = int(key[0]), list(key[1])
        pattern = _rail_pattern(len(text), rails)
        counts = [pattern.count(k) for k in range(rails)]
        rail_str = [""] * rails
        pos = 0
        for k in order:
            rail_str[k] = text[pos: pos + counts[k]]
            pos += counts[k]
        idx = [0] * rails
        out: list[str] = []
        for r in pattern:
            out.append(rail_str[r][idx[r]])
            idx[r] += 1
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any):
        rails = rng.randint(3, 6)
        order = list(range(rails))
        rng.shuffle(order)
        return (rails, tuple(order))

    def describe_key(self, key) -> str:
        return f"redefence rails={int(key[0])} read_order={tuple(key[1])}"

    def key_space_size(self, *, rails: int = 5, **_: Any) -> int:
        return math.factorial(rails)


# ---------------------------------------------------------------------------
# Route transposition (spiral / diagonal)
# ---------------------------------------------------------------------------

def _spiral_coords(rows: int, cols: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    top, bottom, left, right = 0, rows - 1, 0, cols - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            coords.append((top, c))
        top += 1
        for r in range(top, bottom + 1):
            coords.append((r, right))
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                coords.append((bottom, c))
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                coords.append((r, left))
            left += 1
    return coords


def _diagonal_coords(rows: int, cols: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for d in range(rows + cols - 1):
        for r in range(rows):
            c = d - r
            if 0 <= c < cols:
                coords.append((r, c))
    return coords


class RouteCipher(FamilyCipher):
    """Route transposition: fill a grid row-major, read along a route.

    Key is ``{"cols": C, "route": "spiral"|"diagonal"}``.  The grid has ``C``
    columns and ``ceil(n / C)`` rows; the plaintext fills it row-major (the last
    row may be partial).  The route visits every filled cell, so the read-out is
    an exact permutation of the plaintext and round-trips without padding.
    Spiral starts top-left going clockwise inward; diagonal reads successive
    ``r + c`` anti-diagonals.
    """

    name = "route_transposition"

    def _route_indices(self, n: int, key) -> list[int]:
        cols = int(key["cols"])
        route = key.get("route", "spiral")
        rows = math.ceil(n / cols) if n else 0
        coords = _spiral_coords(rows, cols) if route == "spiral" else _diagonal_coords(rows, cols)
        return [r * cols + c for (r, c) in coords if r * cols + c < n]

    def encrypt(self, plaintext: str, key) -> str:
        text = clean(plaintext)
        order = self._route_indices(len(text), key)
        return "".join(text[i] for i in order)

    def decrypt(self, ciphertext: str, key) -> str:
        text = clean(ciphertext)
        order = self._route_indices(len(text), key)
        out = [""] * len(text)
        for pos, src in enumerate(order):
            out[src] = text[pos]
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any):
        return {"cols": rng.randint(4, 8), "route": rng.choice(["spiral", "diagonal"])}

    def describe_key(self, key) -> str:
        return f"route cols={int(key['cols'])} route={key.get('route', 'spiral')}"

    def key_space_size(self, *, max_cols: int = 12, routes: int = 2, **_: Any) -> int:
        return (max_cols - 3) * routes


# ---------------------------------------------------------------------------
# Amsco
# ---------------------------------------------------------------------------

def _amsco_cell_sizes(n: int, ncols: int, start: int) -> list[list[int]]:
    """Per-cell letter counts for an Amsco fill (brick 1/2 alternation)."""
    sizes: list[list[int]] = []
    remaining = n
    row = 0
    while remaining > 0:
        rowsizes: list[int] = []
        for col in range(ncols):
            if remaining <= 0:
                rowsizes.append(0)
                continue
            even = (row + col) % 2 == 0
            base = (1 if even else 2) if start == 1 else (2 if even else 1)
            take = min(base, remaining)
            rowsizes.append(take)
            remaining -= take
        sizes.append(rowsizes)
        row += 1
    return sizes


class AmscoCipher(FamilyCipher):
    """Amsco: incomplete columnar with alternating 1-2 letter cells.

    Key is ``(keyword, start)`` where ``start`` (1 or 2) is the size of the
    top-left cell; sizes alternate in a brick pattern.  Columns are read in
    keyword order.
    """

    name = "amsco"

    def encrypt(self, plaintext: str, key) -> str:
        text = clean(plaintext)
        keyword, start = clean(key[0]), int(key[1])
        ncols = len(keyword)
        sizes = _amsco_cell_sizes(len(text), ncols, start)
        grid: list[list[str]] = []
        idx = 0
        for rowsizes in sizes:
            row: list[str] = []
            for s in rowsizes:
                row.append(text[idx: idx + s])
                idx += s
            grid.append(row)
        out: list[str] = []
        for j in _rank_order(keyword):
            for r in range(len(grid)):
                out.append(grid[r][j])
        return "".join(out)

    def decrypt(self, ciphertext: str, key) -> str:
        text = clean(ciphertext)
        keyword, start = clean(key[0]), int(key[1])
        ncols = len(keyword)
        sizes = _amsco_cell_sizes(len(text), ncols, start)
        nrows = len(sizes)
        col_len = [sum(sizes[r][j] for r in range(nrows)) for j in range(ncols)]
        colstr = [""] * ncols
        idx = 0
        for j in _rank_order(keyword):
            colstr[j] = text[idx: idx + col_len[j]]
            idx += col_len[j]
        grid = [["" for _ in range(ncols)] for _ in range(nrows)]
        for j in range(ncols):
            pos = 0
            for r in range(nrows):
                s = sizes[r][j]
                grid[r][j] = colstr[j][pos: pos + s]
                pos += s
        out: list[str] = []
        for r in range(nrows):
            for j in range(ncols):
                out.append(grid[r][j])
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any):
        return (random_keyword(rng, min_len=5, max_len=8), rng.choice([1, 2]))

    def describe_key(self, key) -> str:
        return f"amsco keyword={clean(key[0])!r} start={int(key[1])}"

    def key_space_size(self, *, columns: int = 7, **_: Any) -> int:
        return math.factorial(columns) * 2


# ---------------------------------------------------------------------------
# Myszkowski
# ---------------------------------------------------------------------------

class MyszkowskiCipher(FamilyCipher):
    """Myszkowski transposition (columnar with repeated key letters).

    Columns whose key letters are equal share a number and are read together
    row by row; unique-numbered columns are read top-to-bottom.  A key with all
    distinct letters degenerates to plain columnar.
    """

    name = "myszkowski"

    def _colnum(self, key: str) -> list[int]:
        distinct = sorted(set(key))
        rank = {ch: i for i, ch in enumerate(distinct)}
        return [rank[ch] for ch in key]

    def encrypt(self, plaintext: str, key: str) -> str:
        text = clean(plaintext)
        key = clean(key)
        ncols = len(key)
        colnum = self._colnum(key)
        cols = [text[j::ncols] for j in range(ncols)]
        out: list[str] = []
        for num in range(max(colnum) + 1):
            group = [j for j in range(ncols) if colnum[j] == num]
            if len(group) == 1:
                out.append(cols[group[0]])
            else:
                maxlen = max(len(cols[j]) for j in group)
                for r in range(maxlen):
                    for j in group:
                        if r < len(cols[j]):
                            out.append(cols[j][r])
        return "".join(out)

    def decrypt(self, ciphertext: str, key: str) -> str:
        text = clean(ciphertext)
        key = clean(key)
        ncols = len(key)
        n = len(text)
        base, extra = divmod(n, ncols)
        col_len = [base + (1 if j < extra else 0) for j in range(ncols)]
        colnum = self._colnum(key)
        cols = [""] * ncols
        idx = 0
        for num in range(max(colnum) + 1):
            group = [j for j in range(ncols) if colnum[j] == num]
            if len(group) == 1:
                j = group[0]
                cols[j] = text[idx: idx + col_len[j]]
                idx += col_len[j]
            else:
                total = sum(col_len[j] for j in group)
                block = text[idx: idx + total]
                idx += total
                buffers: dict[int, list[str]] = {j: [] for j in group}
                bpos = 0
                maxlen = max(col_len[j] for j in group)
                for r in range(maxlen):
                    for j in group:
                        if r < col_len[j]:
                            buffers[j].append(block[bpos])
                            bpos += 1
                for j in group:
                    cols[j] = "".join(buffers[j])
        out: list[str] = []
        maxlen = max(col_len) if col_len else 0
        for r in range(maxlen):
            for j in range(ncols):
                if r < col_len[j]:
                    out.append(cols[j][r])
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any) -> str:
        # small letter pool so repeats occur (exercises the shared-number path)
        pool = "ABCDE"
        length = rng.randint(6, 9)
        return "".join(rng.choice(pool) for _ in range(length))

    def describe_key(self, key: str) -> str:
        k = clean(key)
        return f"myszkowski keyword={k!r} numbers={self._colnum(k)}"

    def key_space_size(self, *, columns: int = 7, **_: Any) -> int:
        return math.factorial(columns)


# ---------------------------------------------------------------------------
# Cadenus
# ---------------------------------------------------------------------------

CADENUS_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVXYZ"  # 25 letters, 'W' omitted (ACA)


class CadenusCipher(FamilyCipher):
    """Cadenus cipher (ACA "Cadenus").

    A 25-row keyed columnar transposition with a per-column cyclic shift.  Our
    25-letter alphabet omits ``W`` (folded to ``V``); the keyword uses those 25
    letters.  Each column is rotated up by the alphabet index of its keyword
    letter, then columns are read in keyword order.  A block is exactly 25 rows;
    the plaintext is padded with ``X`` to a multiple of ``25 * len(keyword)``.
    ``decrypt(encrypt(pt))`` reproduces the padded text (see :meth:`prepare`).
    """

    name = "cadenus"
    ROWS = 25

    def _clean(self, text: str) -> str:
        return clean(text).replace("W", "V")

    def prepare(self, plaintext: str, key: str) -> str:
        text = self._clean(plaintext)
        block = self.ROWS * len(clean(key))
        if len(text) % block:
            text += "X" * (block - len(text) % block)
        return text

    def _shifts(self, key: str) -> list[int]:
        kw = self._clean(key)
        return [CADENUS_ALPHABET.index(ch) for ch in kw]

    def encrypt(self, plaintext: str, key: str) -> str:
        kw = self._clean(key)
        ncols = len(kw)
        text = self.prepare(plaintext, key)
        shifts = self._shifts(key)
        order = _rank_order(kw)
        out: list[str] = []
        block = self.ROWS * ncols
        for b in range(0, len(text), block):
            chunk = text[b: b + block]
            grid = [chunk[r * ncols:(r + 1) * ncols] for r in range(self.ROWS)]
            cols = ["".join(grid[r][j] for r in range(self.ROWS)) for j in range(ncols)]
            rotated = [cols[j][shifts[j]:] + cols[j][:shifts[j]] for j in range(ncols)]
            for j in order:
                out.append(rotated[j])
        return "".join(out)

    def decrypt(self, ciphertext: str, key: str) -> str:
        kw = self._clean(key)
        ncols = len(kw)
        text = clean(ciphertext)
        shifts = self._shifts(key)
        order = _rank_order(kw)
        out: list[str] = []
        block = self.ROWS * ncols
        for b in range(0, len(text), block):
            chunk = text[b: b + block]
            rotated = [""] * ncols
            idx = 0
            for j in order:
                rotated[j] = chunk[idx: idx + self.ROWS]
                idx += self.ROWS
            cols = [rotated[j][self.ROWS - shifts[j]:] + rotated[j][:self.ROWS - shifts[j]]
                    if shifts[j] else rotated[j] for j in range(ncols)]
            for r in range(self.ROWS):
                for j in range(ncols):
                    out.append(cols[j][r])
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any) -> str:
        length = rng.randint(4, 7)
        letters = list(CADENUS_ALPHABET)
        rng.shuffle(letters)
        return "".join(letters[:length])

    def describe_key(self, key: str) -> str:
        kw = self._clean(key)
        return f"cadenus keyword={kw!r} shifts={self._shifts(key)} (25-row block, W->V)"

    def key_space_size(self, *, columns: int = 6, **_: Any) -> int:
        return math.factorial(columns) * (self.ROWS ** columns)


# ---------------------------------------------------------------------------
# Nihilist transposition
# ---------------------------------------------------------------------------

class NihilistTranspositionCipher(FamilyCipher):
    """Nihilist transposition (ACA "Nihilist Transposition").

    An ``n x n`` block (``n = len(keyword)``) filled row-major, with rows AND
    columns permuted by the same keyword-derived permutation, then read
    row-major.  The plaintext is padded with ``X`` to a multiple of ``n^2``;
    ``decrypt(encrypt(pt))`` reproduces the padded text (see :meth:`prepare`).
    """

    name = "nihilist_transposition"

    def prepare(self, plaintext: str, key: str) -> str:
        text = clean(plaintext)
        n = len(clean(key))
        block = n * n
        if len(text) % block:
            text += "X" * (block - len(text) % block)
        return text

    def encrypt(self, plaintext: str, key: str) -> str:
        kw = clean(key)
        n = len(kw)
        perm = _rank_order(kw)
        text = self.prepare(plaintext, key)
        out: list[str] = []
        block = n * n
        for b in range(0, len(text), block):
            chunk = text[b: b + block]
            grid = [chunk[r * n:(r + 1) * n] for r in range(n)]
            for i in range(n):
                for j in range(n):
                    out.append(grid[perm[i]][perm[j]])
        return "".join(out)

    def decrypt(self, ciphertext: str, key: str) -> str:
        kw = clean(key)
        n = len(kw)
        perm = _rank_order(kw)
        text = clean(ciphertext)
        out: list[str] = []
        block = n * n
        for b in range(0, len(text), block):
            chunk = text[b: b + block]
            grid = [["" for _ in range(n)] for _ in range(n)]
            k = 0
            for i in range(n):
                for j in range(n):
                    grid[perm[i]][perm[j]] = chunk[k]
                    k += 1
            for i in range(n):
                for j in range(n):
                    out.append(grid[i][j])
        return "".join(out)

    def random_key(self, rng: random.Random, **_: Any) -> str:
        return random_keyword(rng, min_len=4, max_len=6)

    def describe_key(self, key: str) -> str:
        kw = clean(key)
        return f"nihilist-transposition keyword={kw!r} perm={_rank_order(kw)} (n={len(kw)})"

    def key_space_size(self, *, columns: int = 5, **_: Any) -> int:
        return math.factorial(columns)
