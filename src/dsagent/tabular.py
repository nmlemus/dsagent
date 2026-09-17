"""Reading a table off disk, dispatched on the file's extension.

The harness knows *formats*, not what is in them. Nothing here may grow a notion
of what a column means; it answers two questions and no others: what are the
columns and first rows of this file, and how many rows does it have.

Two callers need that answer for different reasons — `/preview` hands a browser
the head of a file it cannot parse itself, and `show_chart` records how many rows
a chart was drawn from — and neither should have to know that parquet needs
pandas while a CSV needs nothing. A format nobody here can read is a
`TableUnreadable`, which is a true thing to say and better than a guess.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

Table = tuple[list[str], list[list[Any]], int]
"""(columns, rows, total_rows) — `rows` is the head, `total_rows` the whole file."""


class TableUnreadable(RuntimeError):
    """This file is not a table, or nothing installed here can read it."""


def read_table(target: Path, rows: int) -> Table:
    """The first `rows` of a tabular file, plus its column names and row count."""
    reader = READERS.get(target.suffix.lower())
    if reader is None:
        raise TableUnreadable(f"no reader for {target.suffix or 'a file with no extension'}")
    try:
        return reader(target, rows)
    except TableUnreadable:
        raise
    except Exception as e:  # a malformed file is unreadable, not a crash
        raise TableUnreadable(f"cannot read {target.name}: {e}") from e


def count_rows(target: Path) -> int | None:
    """How many rows the file holds, or `None` when that cannot be established.

    `None` rather than 0: a chart drawn from a file this server cannot parse is
    still a chart, and saying "unknown" is not the same as saying "empty".
    """
    try:
        return read_table(target, 1)[2]
    except (TableUnreadable, OSError):
        return None


def _read_parquet(target: Path, rows: int) -> Table:
    """Parquet, through pandas — the reader the cartridge's kernel env installs.

    A server without pandas says so rather than guessing; the file is still
    downloadable, and "this server cannot read that" is a true thing to say.
    """
    try:
        import pandas as pd
    except ImportError:
        raise TableUnreadable("this server has no pandas, so it cannot read parquet") from None

    frame = pd.read_parquet(target)
    head = frame.head(rows)
    return (
        [str(c) for c in head.columns],
        [[jsonable(v) for v in record] for record in head.itertuples(index=False)],
        int(frame.shape[0]),
    )


def _read_delimited(target: Path, rows: int) -> Table:
    """Delimited text, through the stdlib, so a preview needs no dependency.

    `csv` understands quoted fields containing the delimiter, which the browser's
    own split-on-comma does not — and it counts the remaining rows without
    holding the file in memory.
    """
    delimiter = "\t" if target.suffix.lower() == ".tsv" else ","
    with target.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header = next(reader, [])
        table: list[list[Any]] = []
        total = 0
        for row in reader:
            total += 1
            if len(table) < rows:
                table.append(list(row))
    return [str(c) for c in header], table, total


READERS: dict[str, Callable[[Path, int], Table]] = {
    ".parquet": _read_parquet,
    ".pq": _read_parquet,
    ".csv": _read_delimited,
    ".tsv": _read_delimited,
}
"""Extension → reader. The harness knows formats, not what they contain."""


def jsonable(value: Any) -> Any:
    """One cell, as JSON. Anything exotic becomes its own repr rather than a 500."""
    if value is None:
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        # NaN and infinities are not JSON, and a table full of them is exactly
        # what a data-quality preview is for.
        return value if math.isfinite(value) else None
    return str(value)
