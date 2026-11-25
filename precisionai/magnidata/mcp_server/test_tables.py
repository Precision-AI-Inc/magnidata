# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

import pytest

from precisionai.magnidata.mcp_server import tables
from precisionai.magnidata.mcp_server.conftest import TINY_CSV
from precisionai.magnidata.mcp_server.schemas import FilterOp, RowFilter


@pytest.fixture
def table() -> tables.Table:
    return tables.parse_csv(TINY_CSV)


def test_parse_csv_keeps_column_order_and_fills_missing_cells():
    parsed = tables.parse_csv("a,b,c\n1,2\n")
    assert parsed.columns == ["a", "b", "c"]
    assert parsed.rows == [{"a": "1", "b": "2", "c": ""}]


def test_parse_csv_of_empty_text_has_no_columns():
    assert tables.parse_csv("") == tables.Table(columns=[], rows=[])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("640", 640),
        ("-3", -3),
        ("0.5", 0.5),
        ("1e-06", 1e-06),
        ("2.0", 2.0),
        ("cat", "cat"),
        ("nan", "nan"),
    ],
)
def test_cell_value_converts_numbers_and_keeps_text(raw: str, expected: object):
    value = tables.cell_value(raw)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize(("raw", "expected"), [("1.5", 1.5), ("inf", None), ("nan", None), ("x", None), ("", None)])
def test_to_number_rejects_non_finite_and_text(raw: str, expected: float | None):
    assert tables.to_number(raw) == expected


def test_describe_columns_reports_numeric_text_and_empty_columns():
    summaries = {c.name: c for c in tables.describe_columns(tables.parse_csv("n,t,e\n1,x,\n3,x,\n,y,\n"))}
    numeric = summaries["n"]
    assert numeric.kind == "numeric"
    assert numeric.non_empty == 2
    assert (numeric.min, numeric.max, numeric.mean) == (1.0, 3.0, 2.0)
    text = summaries["t"]
    assert text.kind == "text"
    assert text.distinct == 2
    assert [(v.value, v.count) for v in text.top_values or []] == [("x", 2), ("y", 1)]
    assert summaries["e"].kind == "empty"
    assert summaries["e"].non_empty == 0


def test_describe_columns_treats_a_mixed_column_as_text(table: tables.Table):
    summaries = {c.name: c for c in tables.describe_columns(table)}
    assert summaries["note"].kind == "text"
    assert summaries["blur"].kind == "numeric"
    assert summaries["blur"].non_empty == 3


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["width", "image_path"], "image_path"),
        (["thumb", "Image Path"], "Image Path"),
        (["width", "mask_path"], "mask_path"),
        (["first", "second"], "first"),
        ([], None),
    ],
)
def test_image_column_follows_the_api_rule(columns: list[str], expected: str | None):
    assert tables.image_column(columns) == expected


def test_check_columns_names_the_unknown_column(table: tables.Table):
    tables.check_columns(table, ["width", "blur"])
    with pytest.raises(ValueError, match="unknown column 'nope'; columns are: image_path, width"):
        tables.check_columns(table, ["width", "nope"])


def test_check_columns_shortens_a_long_column_list():
    wide = tables.Table(columns=[f"c{i}" for i in range(70)], rows=[])
    with pytest.raises(ValueError, match=r"c59, \.\.\. \(70 in total\)"):
        tables.check_columns(wide, ["nope"])


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        ("==", 640, [0, 2]),
        ("==", "640.0", [0, 2]),
        ("!=", 640, [1, 3]),
        (">", 500, [0, 2, 3]),
        (">=", 640, [0, 2, 3]),
        ("<", 640, [1]),
        ("<=", 640, [0, 1, 2]),
    ],
)
def test_numeric_filters(table: tables.Table, op: FilterOp, value: str | float, expected: list[int]):
    assert tables.select_rows(table, [RowFilter(column="width", op=op, value=value)]) == expected


def test_text_filters(table: tables.Table):
    def select(op: FilterOp, value: str | None = None, column: str = "note") -> list[int]:
        return tables.select_rows(table, [RowFilter(column=column, op=op, value=value)])

    assert select("==", "sharp") == [0, 3]
    assert select("!=", "sharp") == [1, 2]
    assert select("contains", "BLURRY") == [2]
    assert select("empty") == [1]
    assert select("not_empty") == [0, 2, 3]
    assert select(">", "a") == []
    assert select("<", "5", column="blur") == [1]


def test_filters_combine_with_and(table: tables.Table):
    filters = [RowFilter(column="width", op="==", value=640), RowFilter(column="note", op="==", value="sharp")]
    assert tables.select_rows(table, filters) == [0]


def test_sort_puts_empty_cells_last_in_both_directions(table: tables.Table):
    assert tables.select_rows(table, sort_by="blur") == [1, 0, 3, 2]
    assert tables.select_rows(table, sort_by="blur", descending=True) == [3, 0, 1, 2]


def test_sort_orders_numbers_before_text():
    mixed = tables.parse_csv("k,v\n0,b\n1,10\n2,\n3,2\n4,a\n")
    assert tables.select_rows(mixed, sort_by="v") == [3, 1, 4, 0, 2]


def test_select_rows_rejects_unknown_sort_and_filter_columns(table: tables.Table):
    with pytest.raises(ValueError, match="unknown column 'size'"):
        tables.select_rows(table, sort_by="size")
    with pytest.raises(ValueError, match="unknown column 'size'"):
        tables.select_rows(table, [RowFilter(column="size", op="empty")])


def test_row_values_carry_the_row_index(table: tables.Table):
    assert tables.row_values(table, 2, ["width", "blur", "note"]) == {
        "row_index": 2,
        "width": 640,
        "blur": None,
        "note": "Blurry edge",
    }
    assert tables.cells(table, 1, ["blur"]) == {"blur": 3.25}


def test_text_columns_keep_numeric_looking_cells_as_text():
    table = tables.parse_csv("id,code,width\n007,A1,640\n012,42,480\n")
    assert table.numeric_columns == frozenset({"width"})
    assert tables.row_values(table, 0, ["id", "code", "width"]) == {
        "row_index": 0,
        "id": "007",
        "code": "A1",
        "width": 640,
    }
    assert tables.cells(table, 1, ["code"]) == {"code": "42"}
    kinds = {summary.name: summary.kind for summary in tables.describe_columns(table)}
    assert kinds == {"id": "text", "code": "text", "width": "numeric"}


@pytest.mark.parametrize(
    ("raw", "expected"), [("007", False), ("-01", False), ("0", True), ("0.5", True), ("10", True), ("x", False)]
)
def test_is_numeric_text_rejects_zero_padded_ids(raw: str, expected: bool):
    assert tables.is_numeric_text(raw) is expected


def test_cell_value_keeps_text_in_a_text_column():
    assert tables.cell_value("42", numeric=False) == "42"
    assert tables.cell_value("007") == "007"
