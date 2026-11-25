# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Parse, describe, filter, sort and page a dataset's feature table (the CSV the API serves)."""

import csv
import io
import math
import operator
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property

from precisionai.magnidata.mcp_server.schemas import CellValue, ColumnSummary, RowFilter, ValueCount

ROW_INDEX = "row_index"
TOP_VALUES = 5
# A zero-padded integer such as "007" is an identifier, not the number 7.
_ZERO_PADDED = re.compile(r"^[+-]?0\d")

_NUMERIC_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@dataclass(frozen=True)
class Table:
    """A feature table: column names in file order and one dict of raw strings per row."""

    columns: list[str]
    rows: list[dict[str, str]]

    @cached_property
    def numeric_columns(self) -> frozenset[str]:
        """Return the columns whose every non-empty cell is numeric (see `is_numeric_text`)."""
        return frozenset(
            name
            for name in self.columns
            if any(row[name] != "" for row in self.rows)
            and all(is_numeric_text(row[name]) for row in self.rows if row[name] != "")
        )


def parse_csv(text: str) -> Table:
    """Parse CSV text into a `Table`.

    Parameters
    ----------
    text : str
        The CSV, header row first.

    Returns
    -------
    Table
        Columns in file order; a missing cell is an empty string.
    """
    reader = csv.DictReader(io.StringIO(text))
    columns = list(reader.fieldnames or [])
    rows = [{name: row.get(name) or "" for name in columns} for row in reader]
    return Table(columns=columns, rows=rows)


def to_number(value: str) -> float | None:
    """Return `value` as a finite float, or None when it is empty, not a number, NaN or infinite."""
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def is_numeric_text(value: str) -> bool:
    """Return whether `value` is a finite number written as one, not zero-padded text such as the id ``007``."""
    return to_number(value) is not None and not _ZERO_PADDED.match(value.strip())


def cell_value(value: str, numeric: bool = True) -> CellValue:
    """Return a cell as JSON-friendly data: None when empty, int or float in a numeric column, else the text.

    Parameters
    ----------
    value : str
        The raw cell.
    numeric : bool
        Whether the cell's column is numeric; a text column keeps every cell as text, so ``007`` stays ``"007"``.

    Returns
    -------
    CellValue
        None, a number or the original text.
    """
    if value == "":
        return None
    number = to_number(value) if numeric and is_numeric_text(value) else None
    if number is None:
        return value
    return int(number) if number.is_integer() and "." not in value and "e" not in value.lower() else number


def describe_columns(table: Table) -> list[ColumnSummary]:
    """Summarise every column: numeric range and mean, or the most common text values.

    A column is numeric when every non-empty cell is a finite number that is not zero-padded (see
    `is_numeric_text`).

    Parameters
    ----------
    table : Table
        The parsed feature table.

    Returns
    -------
    list of ColumnSummary
        One entry per column, in file order.
    """
    return [
        _describe_column(name, [row[name] for row in table.rows if row[name] != ""], name in table.numeric_columns)
        for name in table.columns
    ]


def _describe_column(name: str, values: list[str], numeric: bool) -> ColumnSummary:
    if not values:
        return ColumnSummary(name=name, kind="empty", non_empty=0)
    if numeric:
        finite = [number for number in map(to_number, values) if number is not None]
        return ColumnSummary(
            name=name,
            kind="numeric",
            non_empty=len(values),
            min=min(finite),
            max=max(finite),
            mean=math.fsum(finite) / len(finite),
        )
    counts = Counter(values)
    return ColumnSummary(
        name=name,
        kind="text",
        non_empty=len(values),
        distinct=len(counts),
        top_values=[ValueCount(value=value, count=count) for value, count in counts.most_common(TOP_VALUES)],
    )


def image_column(columns: Sequence[str]) -> str | None:
    """Return the column holding each row's image path, by the rule the MagniData API itself uses.

    The rule mirrors ``image_path_column`` in ``precisionai/magnidata/training/contract.py``; it is repeated here
    because this package installs without the rest of MagniData.
    """
    for name in columns:
        lowered = name.lower()
        if lowered == "image_path" or ("image" in lowered and "path" in lowered):
            return name
    return next((name for name in columns if "path" in name.lower()), columns[0] if columns else None)


def check_columns(table: Table, names: Iterable[str]) -> None:
    """Raise ValueError naming the first of `names` that is not a column of `table`."""
    known = set(table.columns)
    for name in names:
        if name not in known:
            listed = ", ".join(table.columns[:60])
            more = "" if len(table.columns) <= 60 else f", ... ({len(table.columns)} in total)"
            raise ValueError(f"unknown column {name!r}; columns are: {listed}{more}")


def matches(row: dict[str, str], row_filter: RowFilter) -> bool:
    """Return whether one row satisfies one filter (see `RowFilter.op` for the semantics)."""
    cell = row[row_filter.column]
    op = row_filter.op
    if op == "empty":
        return cell == ""
    if op == "not_empty":
        return cell != ""
    target = "" if row_filter.value is None else str(row_filter.value)
    if op == "contains":
        return target.lower() in cell.lower()
    cell_number, target_number = to_number(cell), to_number(target)
    if cell_number is None or target_number is None:
        return op not in _NUMERIC_OPS and (cell == target) == (op == "==")
    if op in _NUMERIC_OPS:
        return _NUMERIC_OPS[op](cell_number, target_number)
    return (cell_number == target_number) == (op == "==")


def select_rows(
    table: Table, filters: Sequence[RowFilter] = (), sort_by: str | None = None, descending: bool = False
) -> list[int]:
    """Return the indices of the rows that match every filter, in the requested order.

    Parameters
    ----------
    table : Table
        The parsed feature table.
    filters : sequence of RowFilter
        Conditions a row must all satisfy.
    sort_by : str, optional
        Column to order by. Numbers sort numerically and before text; empty cells always come last.
    descending : bool
        Reverse the order (empty cells still come last).

    Returns
    -------
    list of int
        Row indices into ``table.rows``; table order when `sort_by` is None.

    Raises
    ------
    ValueError
        If a filter or `sort_by` names a column the table does not have.
    """
    check_columns(table, [f.column for f in filters] + ([sort_by] if sort_by else []))
    selected = [i for i, row in enumerate(table.rows) if all(matches(row, f) for f in filters)]
    if sort_by is None:
        return selected
    filled = [i for i in selected if table.rows[i][sort_by] != ""]
    blank = [i for i in selected if table.rows[i][sort_by] == ""]
    filled.sort(key=lambda i: _sort_key(table.rows[i][sort_by]), reverse=descending)
    return filled + blank


def _sort_key(value: str) -> tuple[int, float, str]:
    number = to_number(value)
    return (0, number, "") if number is not None else (1, 0.0, value)


def cells(table: Table, index: int, columns: Sequence[str]) -> dict[str, CellValue]:
    """Return the `columns` of one row as `{column: value}`, converted by `cell_value` with the column's type."""
    row = table.rows[index]
    numeric = table.numeric_columns
    return {name: cell_value(row[name], name in numeric) for name in columns}


def row_values(table: Table, index: int, columns: Sequence[str]) -> dict[str, CellValue]:
    """Return one row as `{row_index, column: value, ...}` restricted to `columns`."""
    return {ROW_INDEX: index} | cells(table, index, columns)
