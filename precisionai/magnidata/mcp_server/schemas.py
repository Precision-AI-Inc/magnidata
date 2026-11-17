# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Structured results the MagniData MCP tools return, and the row filter they accept."""

from typing import Literal

from pydantic import BaseModel, Field

CellValue = str | int | float | None
FilterOp = Literal["==", "!=", "<", "<=", ">", ">=", "contains", "empty", "not_empty"]
ImageKind = Literal["image", "mask", "overlay"]
ProjectionMethod = Literal["pca", "tsne", "lle"]


class DatasetInfo(BaseModel):
    """One dataset the MagniData server lists."""

    name: str
    source: str = Field(description="Dataset id. Pass it as `source` to every other tool.")
    description: str = ""
    row_count: int | None = Field(default=None, description="Rows (images) in the feature table.")
    has_embeddings: bool = Field(description="Whether embedding_projection and embedding_clusters work.")
    parent_sources: list[str] = Field(default_factory=list, description="Datasets this one was carved from.")


class DatasetList(BaseModel):
    """Every dataset on the server, with the API address that answered."""

    api_url: str
    datasets: list[DatasetInfo]


class ValueCount(BaseModel):
    """How often one value occurs in a text column."""

    value: str
    count: int


class ColumnSummary(BaseModel):
    """Type and summary statistics of one feature-table column."""

    name: str
    kind: Literal["numeric", "text", "empty"]
    non_empty: int = Field(description="Rows with a value in this column.")
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    distinct: int | None = Field(default=None, description="Distinct values (text columns only).")
    top_values: list[ValueCount] | None = Field(default=None, description="Most common values (text columns only).")


class MaskCoverage(BaseModel):
    """How many of a sample of rows have a segmentation mask."""

    checked: int
    with_masks: int


class DatasetDescription(BaseModel):
    """A dataset's shape: columns with statistics, image column, embeddings and sample images."""

    source: str
    row_count: int
    column_count: int
    image_column: str | None = Field(description="Column holding each row's image path, for get_image.")
    has_embeddings: bool
    mask_coverage: MaskCoverage | None = None
    sample_images: list[str] = Field(default_factory=list, description="Image paths to try with get_image.")
    columns: list[ColumnSummary]


class RowFilter(BaseModel):
    """One condition on a column. Rows must match every filter given."""

    column: str
    op: FilterOp = Field(
        description=(
            "Comparison. `<`, `<=`, `>`, `>=` compare numbers and skip non-numeric cells; `==` and `!=` compare "
            "numbers when both sides are numeric, text otherwise; `contains` is a case-insensitive substring "
            "match; `empty` and `not_empty` ignore `value`."
        )
    )
    value: str | int | float | None = None


class RowPage(BaseModel):
    """One page of feature-table rows, after filtering and sorting."""

    source: str
    total_rows: int
    matching_rows: int
    offset: int
    columns: list[str]
    rows: list[dict[str, CellValue]] = Field(
        description="Each row carries `row_index`, its position in the table, which embedding results use too."
    )


class ProjectionPoint(BaseModel):
    """One row's position in the 3D embedding projection."""

    row_index: int
    x: float
    y: float
    z: float
    values: dict[str, CellValue] | None = None


class Projection(BaseModel):
    """A page of 3D embedding coordinates, one point per table row."""

    source: str
    method: ProjectionMethod
    total_points: int
    offset: int
    points: list[ProjectionPoint]


class ClusterSummary(BaseModel):
    """Size and example rows of one embedding cluster."""

    label: int
    size: int
    example_rows: list[int]
    example_images: list[str] = Field(default_factory=list)


class Clustering(BaseModel):
    """Spherical k-means clusters of a dataset's embeddings."""

    source: str
    k: int
    total_points: int
    clusters: list[ClusterSummary] = Field(description="Largest cluster first.")
    offset: int
    labels: list[int] = Field(description="Cluster label of rows offset, offset + 1, ... in table order.")
