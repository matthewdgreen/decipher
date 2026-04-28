"""Ciphertext order transformers compatible with Zenith-style pipelines.

The first supported target is the subset of Zenith ciphertext transformers
needed for Z340-style replay: range reversal, one-character shifts, locked
ranges, keyed column transposition, N-down/M-across grid walks, and simple
grid route reads for bounded transform search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransformStep:
    """One Zenith-compatible transformation step."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "TransformStep":
        name = raw.get("name") or raw.get("transformerName") or raw.get("transformer_name")
        if not name:
            raise ValueError("transform step must include name/transformerName")
        data = raw.get("data")
        if data is None:
            data = {
                k: v
                for k, v in raw.items()
                if k not in {"name", "transformerName", "transformer_name"}
            }
        if not isinstance(data, dict):
            raise ValueError("transform step data must be an object")
        return cls(name=str(name), data=dict(data))

    def to_raw(self) -> dict[str, Any]:
        return {"name": self.name, "data": dict(self.data)}


@dataclass(frozen=True)
class TransformPipeline:
    """Ordered transformer pipeline with optional grid dimensions."""

    steps: tuple[TransformStep, ...] = ()
    columns: int | None = None
    rows: int | None = None

    @classmethod
    def from_raw(cls, raw: Any) -> "TransformPipeline | None":
        if raw in (None, [], {}):
            return None
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, list):
            return cls(steps=tuple(TransformStep.from_raw(step) for step in raw))
        if not isinstance(raw, dict):
            raise ValueError("transform pipeline must be an object or list")
        raw_steps = raw.get("steps") or raw.get("appliedCiphertextTransformers") or []
        if not isinstance(raw_steps, list):
            raise ValueError("transform pipeline steps must be a list")
        columns = raw.get("columns") or raw.get("grid_columns")
        rows = raw.get("rows") or raw.get("grid_rows")
        return cls(
            steps=tuple(TransformStep.from_raw(step) for step in raw_steps),
            columns=int(columns) if columns is not None else None,
            rows=int(rows) if rows is not None else None,
        )

    def to_raw(self) -> dict[str, Any]:
        out: dict[str, Any] = {"steps": [step.to_raw() for step in self.steps]}
        if self.columns is not None:
            out["columns"] = self.columns
        if self.rows is not None:
            out["rows"] = self.rows
        return out

    def is_empty(self) -> bool:
        return not self.steps


@dataclass(frozen=True)
class TransformResult:
    tokens: list[int]
    locked: list[bool]
    pipeline: TransformPipeline


def apply_transform_pipeline(
    tokens: list[int],
    pipeline: TransformPipeline | dict[str, Any] | list[dict[str, Any]] | None,
) -> TransformResult:
    """Apply a Zenith-compatible pipeline to a token stream.

    Ranges are inclusive and zero-based, matching Zenith's Java transformers.
    ``LockCharacters`` marks positions so later grid-walk transforms skip those
    characters and append them at the end of the transformed row range.
    """

    parsed = TransformPipeline.from_raw(pipeline) or TransformPipeline()
    current = list(tokens)
    locked = [False] * len(current)
    for step in parsed.steps:
        current, locked = _apply_step(current, locked, parsed, step)
    return TransformResult(tokens=current, locked=locked, pipeline=parsed)


def make_inverse_input_for_pipeline(
    target_tokens: list[int],
    pipeline: TransformPipeline | dict[str, Any] | list[dict[str, Any]] | None,
) -> list[int]:
    """Return an input stream that the pipeline transforms into target_tokens."""

    parsed = TransformPipeline.from_raw(pipeline) or TransformPipeline()
    order = apply_transform_pipeline(list(range(len(target_tokens))), parsed).tokens
    if sorted(order) != list(range(len(target_tokens))):
        raise ValueError("transform pipeline did not produce a permutation")
    scrambled = [0] * len(target_tokens)
    for output_index, input_index in enumerate(order):
        scrambled[input_index] = target_tokens[output_index]
    return scrambled


def candidate_transform_pipelines(
    *,
    token_count: int,
    columns: int | None = None,
    profile: str = "small",
) -> list[TransformPipeline]:
    """Small bounded candidate set for first-pass transform search."""

    candidates = [TransformPipeline()]
    candidates.append(TransformPipeline(steps=(TransformStep("Reverse", {}),), columns=columns))
    if token_count > 1:
        candidates.append(
            TransformPipeline(
                steps=(TransformStep("ShiftCharactersLeft", {"rangeStart": 0, "rangeEnd": token_count - 1}),),
                columns=columns,
            )
        )
        candidates.append(
            TransformPipeline(
                steps=(TransformStep("ShiftCharactersRight", {"rangeStart": 0, "rangeEnd": token_count - 1}),),
                columns=columns,
            )
        )
    if columns and columns > 1 and token_count >= columns * 2:
        rows = token_count // columns
        candidates.append(
            TransformPipeline(
                steps=(TransformStep("NDownMAcross", {"rangeStart": 0, "rangeEnd": rows - 1, "down": 1, "across": 1}),),
                columns=columns,
                rows=rows,
            )
        )
    if profile != "small":
        for width in range(2, min(8, token_count)):
            if token_count % width == 0:
                candidates.append(
                    TransformPipeline(
                        steps=(TransformStep("UnwrapTransposition", {"key": "".join(chr(97 + i) for i in range(width))}),),
                        columns=width,
                        rows=token_count // width,
                    )
                )
    return candidates


def _apply_step(
    tokens: list[int],
    locked: list[bool],
    pipeline: TransformPipeline,
    step: TransformStep,
) -> tuple[list[int], list[bool]]:
    name = _canonical_name(step.name)
    data = step.data
    if name == "reverse":
        start, end = _range(data, len(tokens))
        new_tokens = list(tokens)
        new_locked = list(locked)
        if start <= end:
            new_tokens[start : end + 1] = list(reversed(tokens[start : end + 1]))
            new_locked[start : end + 1] = list(reversed(locked[start : end + 1]))
        return new_tokens, new_locked
    if name == "shiftcharactersleft":
        start, end = _range(data, len(tokens))
        return _shift(tokens, locked, start, end, -1)
    if name == "shiftcharactersright":
        start, end = _range(data, len(tokens))
        return _shift(tokens, locked, start, end, 1)
    if name == "lockcharacters":
        start, end = _range(data, len(tokens))
        new_locked = list(locked)
        for i in range(start, end + 1):
            new_locked[i] = True
        return list(tokens), new_locked
    if name == "ndownmacross":
        return _n_down_m_across(tokens, locked, pipeline, data)
    if name == "transposition":
        return _columnar_transposition(tokens, locked, data, unwrap=False)
    if name == "unwraptransposition":
        return _columnar_transposition(tokens, locked, data, unwrap=True)
    if name == "routeread":
        return _route_read(tokens, locked, pipeline, data)
    if name == "splitgridroute":
        return _split_grid_route(tokens, locked, pipeline, data)
    if name == "gridpermute":
        return _grid_permute(tokens, locked, pipeline, data)
    raise ValueError(f"unsupported ciphertext transformer: {step.name}")


def _canonical_name(name: str) -> str:
    return re_sub_non_alnum(name).lower()


def re_sub_non_alnum(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum())


def _range(data: dict[str, Any], length: int) -> tuple[int, int]:
    if length <= 0:
        return 0, -1
    start = max(int(data.get("rangeStart", 0)), 0)
    end = min(int(data.get("rangeEnd", length - 1)), length - 1)
    return start, end


def _shift(
    tokens: list[int],
    locked: list[bool],
    start: int,
    end: int,
    direction: int,
) -> tuple[list[int], list[bool]]:
    new_tokens = list(tokens)
    new_locked = list(locked)
    if start > end:
        return new_tokens, new_locked
    token_range = tokens[start : end + 1]
    lock_range = locked[start : end + 1]
    if direction < 0:
        token_range = token_range[1:] + token_range[:1]
        lock_range = lock_range[1:] + lock_range[:1]
    else:
        token_range = token_range[-1:] + token_range[:-1]
        lock_range = lock_range[-1:] + lock_range[:-1]
    new_tokens[start : end + 1] = token_range
    new_locked[start : end + 1] = lock_range
    return new_tokens, new_locked


def _n_down_m_across(
    tokens: list[int],
    locked: list[bool],
    pipeline: TransformPipeline,
    data: dict[str, Any],
) -> tuple[list[int], list[bool]]:
    columns = pipeline.columns
    if columns is None or columns <= 0:
        raise ValueError("NDownMAcross requires pipeline columns")
    rows = pipeline.rows or (len(tokens) // columns)
    if rows <= 0:
        return list(tokens), list(locked)
    row_start = max(int(data.get("rangeStart", 0)), 0)
    row_end = min(int(data.get("rangeEnd", rows - 1)), rows - 1)
    if row_start > row_end:
        return list(tokens), list(locked)

    down = int(data["down"])
    across = int(data["across"])
    char_start = row_start * columns
    char_end = min((row_end + 1) * columns, len(tokens))
    if char_start >= char_end:
        return list(tokens), list(locked)

    new_tokens = list(tokens)
    new_locked = list(locked)
    output = char_start
    cursor = char_start
    append_later: list[int] = []
    offset = down * columns + across
    for _ in range(char_start, char_end):
        if not locked[cursor]:
            new_tokens[output] = tokens[cursor]
            new_locked[output] = locked[cursor]
            output += 1
        else:
            append_later.append(cursor)
        previous = cursor
        cursor += offset
        if cursor - across >= char_end:
            cursor = char_start + (cursor % columns)
        elif (previous % columns) > (cursor % columns):
            cursor -= columns

    tail_start = char_end - len(append_later)
    for i, source_index in enumerate(append_later):
        new_tokens[tail_start + i] = tokens[source_index]
        new_locked[tail_start + i] = locked[source_index]
    return new_tokens, new_locked


def _columnar_transposition(
    tokens: list[int],
    locked: list[bool],
    data: dict[str, Any],
    *,
    unwrap: bool,
) -> tuple[list[int], list[bool]]:
    key = str(data.get("key") or data.get("argument") or "")
    indices = _indices_for_key(key)
    if len(indices) < 2 or len(indices) >= len(tokens):
        raise ValueError("transposition key length must be greater than 1 and less than token count")
    rows = len(tokens) // len(indices)
    usable = rows * len(indices)
    new_tokens = list(tokens)
    new_locked = list(locked)
    out = 0
    for i in range(len(indices)):
        column_index = indices.index(i)
        for row in range(rows):
            source = row * len(indices) + column_index
            if unwrap:
                target = source
                source = out
            else:
                target = out
            new_tokens[target] = tokens[source]
            new_locked[target] = locked[source]
            out += 1
    if usable < len(tokens):
        new_tokens[usable:] = tokens[usable:]
        new_locked[usable:] = locked[usable:]
    return new_tokens, new_locked


def _route_read(
    tokens: list[int],
    locked: list[bool],
    pipeline: TransformPipeline,
    data: dict[str, Any],
) -> tuple[list[int], list[bool]]:
    columns = int(data.get("columns") or pipeline.columns or 0)
    rows = int(data.get("rows") or pipeline.rows or 0)
    if columns <= 0:
        raise ValueError("RouteRead requires columns")
    if rows <= 0:
        rows = len(tokens) // columns
    usable = min(rows * columns, len(tokens))
    if rows <= 0 or usable <= 0:
        return list(tokens), list(locked)
    route = str(data.get("route") or "columns_down").lower()
    if route == "offset_chain":
        positions = _offset_chain_positions(rows, columns, int(data.get("step", 1)))
    elif route in {"rows_progressive_shift", "columns_progressive_shift"}:
        positions = _progressive_shift_positions(
            rows,
            columns,
            route=route,
            shift=int(data.get("shift", 1)),
        )
    else:
        positions = _route_positions(rows, columns, route)
    if len(positions) != rows * columns:
        raise ValueError(f"route {route!r} did not cover the grid")
    order = [row * columns + col for row, col in positions if row * columns + col < usable]
    if sorted(order) != list(range(usable)):
        raise ValueError(f"route {route!r} did not produce a grid permutation")
    new_tokens = [tokens[i] for i in order] + list(tokens[usable:])
    new_locked = [locked[i] for i in order] + list(locked[usable:])
    return new_tokens, new_locked


def _split_grid_route(
    tokens: list[int],
    locked: list[bool],
    pipeline: TransformPipeline,
    data: dict[str, Any],
) -> tuple[list[int], list[bool]]:
    columns = int(data.get("columns") or pipeline.columns or 0)
    rows = int(data.get("rows") or pipeline.rows or 0)
    if columns <= 1:
        raise ValueError("SplitGridRoute requires columns > 1")
    if rows <= 0:
        rows = len(tokens) // columns
    usable = min(rows * columns, len(tokens))
    if rows <= 1 or usable <= 0:
        return list(tokens), list(locked)
    orientation = str(data.get("orientation") or "horizontal").lower()
    split = int(data.get("split", rows // 2 if orientation == "horizontal" else columns // 2))
    first_route = str(data.get("firstRoute") or data.get("first_route") or "rows").lower()
    second_route = str(data.get("secondRoute") or data.get("second_route") or "rows").lower()
    region_order = str(data.get("regionOrder") or data.get("region_order") or "normal").lower()

    if orientation == "horizontal":
        if split <= 0 or split >= rows:
            raise ValueError("horizontal SplitGridRoute split must be inside row range")
        first = _subgrid_order(0, 0, split, columns, columns, first_route)
        second = _subgrid_order(split, 0, rows - split, columns, columns, second_route)
    elif orientation == "vertical":
        if split <= 0 or split >= columns:
            raise ValueError("vertical SplitGridRoute split must be inside column range")
        first = _subgrid_order(0, 0, rows, split, columns, first_route)
        second = _subgrid_order(0, split, rows, columns - split, columns, second_route)
    else:
        raise ValueError("SplitGridRoute orientation must be horizontal or vertical")

    order = (second + first) if region_order == "swap" else (first + second)
    if sorted(order) != list(range(usable)):
        raise ValueError("SplitGridRoute did not produce a grid permutation")
    new_tokens = [tokens[i] for i in order] + list(tokens[usable:])
    new_locked = [locked[i] for i in order] + list(locked[usable:])
    return new_tokens, new_locked


def _grid_permute(
    tokens: list[int],
    locked: list[bool],
    pipeline: TransformPipeline,
    data: dict[str, Any],
) -> tuple[list[int], list[bool]]:
    columns = int(data.get("columns") or pipeline.columns or 0)
    rows = int(data.get("rows") or pipeline.rows or 0)
    if columns <= 1:
        raise ValueError("GridPermute requires columns > 1")
    if rows <= 0:
        rows = len(tokens) // columns
    usable = min(rows * columns, len(tokens))
    if rows <= 1 or usable <= 0:
        return list(tokens), list(locked)
    row_order = _grid_order(data.get("rowOrder") or data.get("row_order") or "identity", rows)
    column_order = _grid_order(data.get("columnOrder") or data.get("column_order") or "identity", columns)
    order = [
        row * columns + col
        for row in row_order
        for col in column_order
        if row * columns + col < usable
    ]
    if sorted(order) != list(range(usable)):
        raise ValueError("GridPermute did not produce a grid permutation")
    new_tokens = [tokens[i] for i in order] + list(tokens[usable:])
    new_locked = [locked[i] for i in order] + list(locked[usable:])
    return new_tokens, new_locked


def _grid_order(raw: Any, size: int) -> list[int]:
    if isinstance(raw, list):
        order = [int(item) for item in raw]
        if sorted(order) != list(range(size)):
            raise ValueError("explicit grid order is not a permutation")
        return order
    name = str(raw or "identity").lower()
    if name in {"identity", "normal", "rows", "columns"}:
        return list(range(size))
    if name in {"reverse", "reversed"}:
        return list(range(size - 1, -1, -1))
    if name == "even_odd":
        return list(range(0, size, 2)) + list(range(1, size, 2))
    if name == "odd_even":
        return list(range(1, size, 2)) + list(range(0, size, 2))
    if name == "outside_in":
        return _outside_in_order(size)
    if name == "inside_out":
        return list(reversed(_outside_in_order(size)))
    raise ValueError(f"unsupported grid order: {raw}")


def _outside_in_order(size: int) -> list[int]:
    order: list[int] = []
    left = 0
    right = size - 1
    while left <= right:
        order.append(left)
        if left != right:
            order.append(right)
        left += 1
        right -= 1
    return order


def _subgrid_order(
    row_offset: int,
    col_offset: int,
    rows: int,
    columns: int,
    full_columns: int,
    route: str,
) -> list[int]:
    positions = _route_positions(rows, columns, route)
    return [
        (row_offset + row) * full_columns + col_offset + col
        for row, col in positions
    ]


def _route_positions(rows: int, columns: int, route: str) -> list[tuple[int, int]]:
    if route in {"rows", "rows_right", "rows_ltr"}:
        return [(r, c) for r in range(rows) for c in range(columns)]
    if route in {"rows_reverse", "rows_left", "rows_rtl"}:
        return [(r, c) for r in range(rows) for c in range(columns - 1, -1, -1)]
    if route in {"rows_boustrophedon", "boustrophedon_rows"}:
        return [
            (r, c)
            for r in range(rows)
            for c in (range(columns) if r % 2 == 0 else range(columns - 1, -1, -1))
        ]
    if route in {"columns", "columns_down", "columns_ttb"}:
        return [(r, c) for c in range(columns) for r in range(rows)]
    if route in {"columns_up", "columns_btt"}:
        return [(r, c) for c in range(columns) for r in range(rows - 1, -1, -1)]
    if route in {"columns_boustrophedon", "boustrophedon_columns"}:
        return [
            (r, c)
            for c in range(columns)
            for r in (range(rows) if c % 2 == 0 else range(rows - 1, -1, -1))
        ]
    if route == "spiral_clockwise":
        return _spiral_positions(rows, columns, clockwise=True)
    if route == "spiral_counterclockwise":
        return _spiral_positions(rows, columns, clockwise=False)
    if route == "diagonal_down_right":
        return _diagonal_positions(rows, columns, down=True, right=True)
    if route == "diagonal_down_left":
        return _diagonal_positions(rows, columns, down=True, right=False)
    if route == "diagonal_up_right":
        return _diagonal_positions(rows, columns, down=False, right=True)
    if route == "diagonal_up_left":
        return _diagonal_positions(rows, columns, down=False, right=False)
    if route == "diagonal_zigzag_down_right":
        return _diagonal_zigzag_positions(rows, columns, down=True, right=True)
    if route == "diagonal_zigzag_down_left":
        return _diagonal_zigzag_positions(rows, columns, down=True, right=False)
    if route == "checkerboard_even_odd":
        return _checkerboard_positions(rows, columns, even_first=True)
    if route == "checkerboard_odd_even":
        return _checkerboard_positions(rows, columns, even_first=False)
    if route == "row_column_interleave":
        return _interleave_orders(
            _route_positions(rows, columns, "rows"),
            _route_positions(rows, columns, "columns_down"),
        )
    if route == "column_row_interleave":
        return _interleave_orders(
            _route_positions(rows, columns, "columns_down"),
            _route_positions(rows, columns, "rows"),
        )
    raise ValueError(f"unsupported RouteRead route: {route}")


def _spiral_positions(rows: int, columns: int, *, clockwise: bool) -> list[tuple[int, int]]:
    top = 0
    bottom = rows - 1
    left = 0
    right = columns - 1
    out: list[tuple[int, int]] = []
    while top <= bottom and left <= right:
        if clockwise:
            out.extend((top, c) for c in range(left, right + 1))
            top += 1
            out.extend((r, right) for r in range(top, bottom + 1))
            right -= 1
            if top <= bottom:
                out.extend((bottom, c) for c in range(right, left - 1, -1))
                bottom -= 1
            if left <= right:
                out.extend((r, left) for r in range(bottom, top - 1, -1))
                left += 1
        else:
            out.extend((r, left) for r in range(top, bottom + 1))
            left += 1
            out.extend((bottom, c) for c in range(left, right + 1))
            bottom -= 1
            if left <= right:
                out.extend((r, right) for r in range(bottom, top - 1, -1))
                right -= 1
            if top <= bottom:
                out.extend((top, c) for c in range(right, left - 1, -1))
                top += 1
    return out


def _diagonal_positions(rows: int, columns: int, *, down: bool, right: bool) -> list[tuple[int, int]]:
    row_order = list(range(rows)) if down else list(range(rows - 1, -1, -1))
    col_order = list(range(columns)) if right else list(range(columns - 1, -1, -1))
    row_rank = {row: i for i, row in enumerate(row_order)}
    col_rank = {col: i for i, col in enumerate(col_order)}
    out: list[tuple[int, int]] = []
    for diagonal in range(rows + columns - 1):
        cells = [
            (row, col)
            for row in row_order
            for col in col_order
            if row_rank[row] + col_rank[col] == diagonal
        ]
        out.extend(cells)
    return out


def _diagonal_zigzag_positions(rows: int, columns: int, *, down: bool, right: bool) -> list[tuple[int, int]]:
    row_order = list(range(rows)) if down else list(range(rows - 1, -1, -1))
    col_order = list(range(columns)) if right else list(range(columns - 1, -1, -1))
    row_rank = {row: i for i, row in enumerate(row_order)}
    col_rank = {col: i for i, col in enumerate(col_order)}
    out: list[tuple[int, int]] = []
    for diagonal in range(rows + columns - 1):
        cells = [
            (row, col)
            for row in row_order
            for col in col_order
            if row_rank[row] + col_rank[col] == diagonal
        ]
        if diagonal % 2:
            cells.reverse()
        out.extend(cells)
    return out


def _checkerboard_positions(rows: int, columns: int, *, even_first: bool) -> list[tuple[int, int]]:
    parities = (0, 1) if even_first else (1, 0)
    return [
        (r, c)
        for parity in parities
        for r in range(rows)
        for c in range(columns)
        if (r + c) % 2 == parity
    ]


def _interleave_orders(
    first: list[tuple[int, int]],
    second: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i in range(max(len(first), len(second))):
        for order in (first, second):
            if i >= len(order):
                continue
            cell = order[i]
            if cell in seen:
                continue
            seen.add(cell)
            out.append(cell)
    return out


def _progressive_shift_positions(
    rows: int,
    columns: int,
    *,
    route: str,
    shift: int,
) -> list[tuple[int, int]]:
    if route == "rows_progressive_shift":
        return [
            (r, (c + r * shift) % columns)
            for r in range(rows)
            for c in range(columns)
        ]
    return [
        ((r + c * shift) % rows, c)
        for c in range(columns)
        for r in range(rows)
    ]


def _offset_chain_positions(rows: int, columns: int, step: int) -> list[tuple[int, int]]:
    size = rows * columns
    if size <= 0:
        return []
    step = step % size
    if step <= 0:
        step = 1
    if math_gcd(step, size) != 1:
        raise ValueError("offset_chain step must be coprime with grid size")
    return [(index // columns, index % columns) for index in ((i * step) % size for i in range(size))]


def math_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _indices_for_key(key: str) -> list[int]:
    if not key:
        return []
    key = key.lower()
    next_index = 0
    indices: list[int | None] = [None] * len(key)
    for letter_ord in range(ord("a"), ord("z") + 1):
        letter = chr(letter_ord)
        for pos, ch in enumerate(key):
            if ch == letter:
                indices[pos] = next_index
                next_index += 1
    if any(index is None for index in indices):
        raise ValueError("transposition key must contain only letters")
    return [int(index) for index in indices]
