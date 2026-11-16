# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""MCP server exposing MagniData's datasets, features, embeddings and images as read-only tools.

The server is a thin client of MagniData's REST API: it holds no data of its own and never calls an endpoint that
changes data, builds datasets or trains. It runs over stdio for a local MCP client, or over Streamable HTTP when
started with ``--transport streamable-http``.
"""

import argparse
import logging
import os
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Annotated

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.applications import Starlette

from precisionai.magnidata.mcp_server import tables
from precisionai.magnidata.mcp_server.client import (
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT_SECONDS,
    DEMO_API_URL,
    PACKAGE_VERSION,
    MagniDataAPIError,
    MagniDataClient,
)
from precisionai.magnidata.mcp_server.schemas import (
    Clustering,
    ClusterSummary,
    DatasetDescription,
    DatasetInfo,
    DatasetList,
    ImageKind,
    MaskCoverage,
    Projection,
    ProjectionMethod,
    ProjectionPoint,
    RowFilter,
    RowPage,
)

API_URL_ENV = "MAGNIDATA_API_URL"
REPOSITORY_URL = "https://github.com/Precision-AI-Inc/magnidata"
HTTP_PATH = "/mcp"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")

INSTRUCTIONS = """\
MagniData explores image datasets (for example segmentation datasets): a table of per-image features such as \
size, blur, exposure, colour and annotation coverage, image embeddings with 3D projections and clusters, and the \
images with their masks. Every tool only reads.

Start with list_datasets and pass a dataset's `source` to the other tools. describe_dataset gives the columns and \
their statistics; query_rows filters and sorts the table; embedding_projection and embedding_clusters work when \
`has_embeddings` is true and return one entry per table row, matched by `row_index`; get_image shows an image path \
from the dataset's image column, or its mask or overlay."""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

Source = Annotated[str, Field(description="Dataset `source` from list_datasets, such as data_user/coco128.csv.")]
Variant = Annotated[
    str | None, Field(description="Comparison embedding model from describe_dataset; omit for the dataset's own.")
]
Offset = Annotated[int, Field(ge=0, description="Index of the first row to return.")]


@contextmanager
def _as_tool_error() -> Iterator[None]:
    """Report API failures and bad arguments to the client as tool errors, with their message."""
    try:
        yield
    except (MagniDataAPIError, ValueError) as err:
        raise ToolError(str(err)) from err


class MagniDataTools:
    """The tool implementations, bound to one API client.

    Parameters
    ----------
    client : MagniDataClient
        Client for the MagniData API the tools read from.
    """

    def __init__(self, client: MagniDataClient) -> None:
        self.client = client

    async def list_datasets(self) -> DatasetList:
        """List every dataset on the MagniData server."""
        with _as_tool_error():
            records = await self.client.list_datasets()
        datasets = [
            DatasetInfo(
                name=str(record.get("name") or record.get("source", "")),
                source=str(record.get("source", "")),
                description=str(record.get("description") or ""),
                row_count=record.get("row_count"),
                has_embeddings=bool(record.get("has_embeddings")),
                parent_sources=list(record.get("parent_sources") or []),
            )
            for record in records
        ]
        return DatasetList(api_url=self.client.api_url, datasets=datasets)

    async def describe_dataset(self, source: Source) -> DatasetDescription:
        """Describe one dataset: its columns with statistics, image column, embeddings and sample images."""
        with _as_tool_error():
            summary = await self.client.dataset_summary(source)
            table = tables.parse_csv(await self.client.dataset_csv(source))
            has_embeddings = bool(summary.get("has_embeddings"))
            variants = await self.client.embedding_variants(source) if has_embeddings else []
        coverage = summary.get("mask_coverage")
        return DatasetDescription(
            source=source,
            row_count=len(table.rows),
            column_count=len(table.columns),
            image_column=summary.get("image_column") or tables.image_column(table.columns),
            has_embeddings=has_embeddings,
            embedding_variants=variants,
            mask_coverage=MaskCoverage(**coverage) if coverage else None,
            sample_images=list(summary.get("sample_images") or []),
            columns=tables.describe_columns(table),
        )

    async def query_rows(
        self,
        source: Source,
        columns: Annotated[list[str] | None, Field(description="Columns to return; omit for all of them.")] = None,
        filters: Annotated[
            list[RowFilter] | None, Field(description="Conditions every returned row must meet.")
        ] = None,
        sort_by: Annotated[str | None, Field(description="Column to sort by; omit to keep table order.")] = None,
        descending: bool = False,
        limit: Annotated[int, Field(ge=1, le=200, description="Rows to return.")] = 20,
        offset: Offset = 0,
    ) -> RowPage:
        """Filter, sort and page a dataset's feature table."""
        with _as_tool_error():
            table = tables.parse_csv(await self.client.dataset_csv(source))
            selected_columns = list(columns) if columns else table.columns
            tables.check_columns(table, selected_columns)
            selected = tables.select_rows(table, filters or [], sort_by, descending)
        page = selected[offset : offset + limit]
        return RowPage(
            source=source,
            total_rows=len(table.rows),
            matching_rows=len(selected),
            offset=offset,
            columns=selected_columns,
            rows=[tables.row_values(table, index, selected_columns) for index in page],
        )

    async def embedding_projection(
        self,
        source: Source,
        method: Annotated[
            ProjectionMethod,
            Field(description="pca is fast; tsne and lle separate local neighbourhoods but can take a while."),
        ] = "pca",
        variant: Variant = None,
        include_columns: Annotated[
            list[str] | None,
            Field(description="Table columns to attach to each point, such as the image column or a label."),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=2000, description="Points to return.")] = 200,
        offset: Offset = 0,
    ) -> Projection:
        """Return 3D coordinates of a dataset's image embeddings, one point per table row."""
        with _as_tool_error():
            positions = await self.client.embedding_projection(source, method, variant)
            table = await self._table_for(source, include_columns)
        points = [
            ProjectionPoint(
                row_index=index,
                x=round(position[0], 4),
                y=round(position[1], 4),
                z=round(position[2], 4),
                values=tables.cells(table, index, include_columns or []) if table else None,
            )
            for index, position in enumerate(positions[offset : offset + limit], start=offset)
        ]
        return Projection(
            source=source, method=method, variant=variant, total_points=len(positions), offset=offset, points=points
        )

    async def embedding_clusters(
        self,
        source: Source,
        k: Annotated[int, Field(ge=1, le=50, description="Number of clusters.")] = 8,
        variant: Variant = None,
        examples_per_cluster: Annotated[
            int, Field(ge=0, le=10, description="Example rows and images listed per cluster.")
        ] = 3,
        limit: Annotated[int, Field(ge=0, le=5000, description="Per-row labels to return.")] = 500,
        offset: Offset = 0,
    ) -> Clustering:
        """Cluster a dataset's embeddings with spherical k-means and summarise each cluster."""
        with _as_tool_error():
            labels = await self.client.embedding_clusters(source, k, variant)
            table = tables.parse_csv(await self.client.dataset_csv(source)) if examples_per_cluster else None
        image_column = tables.image_column(table.columns) if table else None
        clusters = []
        for label, size in Counter(labels).most_common():
            rows = [i for i, row_label in enumerate(labels) if row_label == label][:examples_per_cluster]
            images = (
                [table.rows[i][image_column] for i in rows if i < len(table.rows)] if table and image_column else []
            )
            clusters.append(ClusterSummary(label=label, size=size, example_rows=rows, example_images=images))
        return Clustering(
            source=source,
            k=k,
            variant=variant,
            total_points=len(labels),
            clusters=clusters,
            offset=offset,
            labels=labels[offset : offset + limit],
        )

    async def get_image(
        self,
        path: Annotated[str, Field(description="Image path from the dataset's image column or sample_images.")],
        kind: Annotated[
            ImageKind,
            Field(description="image, its segmentation mask, or the mask drawn over the image (overlay)."),
        ] = "image",
        max_size: Annotated[int, Field(ge=64, le=1600, description="Longest side in pixels.")] = 512,
        alpha: Annotated[float, Field(ge=0.0, le=1.0, description="Mask opacity for an overlay.")] = 0.5,
    ) -> Image:
        """Return an image from a dataset, its mask or an overlay, scaled down to a thumbnail by default."""
        with _as_tool_error():
            data, media_type = await self.client.image(path, kind, max_size, alpha)
        return Image(data=data, format=media_type.removeprefix("image/"))

    async def _table_for(self, source: str, columns: Sequence[str] | None) -> tables.Table | None:
        """Fetch the feature table when `columns` asks for values from it, and check those columns exist."""
        if not columns:
            return None
        table = tables.parse_csv(await self.client.dataset_csv(source))
        tables.check_columns(table, columns)
        return table


def create_server(client: MagniDataClient) -> MCPServer:
    """Build the MCP server with every MagniData tool registered.

    Parameters
    ----------
    client : MagniDataClient
        Client for the MagniData API the tools read from.

    Returns
    -------
    MCPServer
        A server ready to run over stdio or to serve as an HTTP app.
    """
    server = MCPServer(
        name="magnidata",
        title="MagniData",
        description="Read-only access to MagniData image datasets: features, embeddings, clusters and images.",
        instructions=INSTRUCTIONS,
        website_url=REPOSITORY_URL,
        version=PACKAGE_VERSION,
    )
    tools = MagniDataTools(client)
    registrations = (
        (
            tools.list_datasets,
            "List datasets",
            "List the datasets on the MagniData server, with row counts and "
            "whether embeddings exist. Start here: other tools take a dataset's `source`.",
        ),
        (
            tools.describe_dataset,
            "Describe dataset",
            "Describe a dataset's feature table: every column with its "
            "type and statistics (numeric range and mean, or common text values), the image column, mask coverage, "
            "embedding variants and sample image paths.",
        ),
        (
            tools.query_rows,
            "Query rows",
            "Filter, sort and page a dataset's per-image feature table, such as the "
            "blurriest images or those with an annotation ratio above 0.5. Each row carries its `row_index`.",
        ),
        (
            tools.embedding_projection,
            "Embedding projection",
            "Project a dataset's image embeddings to 3D (PCA, "
            "t-SNE or LLE) and return a page of points, one per table row, optionally with table columns attached.",
        ),
        (
            tools.embedding_clusters,
            "Embedding clusters",
            "Cluster a dataset's image embeddings into k groups "
            "(spherical k-means on the full vectors) and return each cluster's size and example images, plus the "
            "per-row labels.",
        ),
        (
            tools.get_image,
            "Get image",
            "Fetch an image from a dataset as image content, or its segmentation mask, "
            "or the mask drawn over it. Thumbnail size (512 px) unless max_size says otherwise.",
        ),
    )
    for fn, title, description in registrations:
        server.add_tool(fn, title=title, description=description, annotations=READ_ONLY)
    return server


def transport_security(
    allowed_hosts: Sequence[str] = (), allowed_origins: Sequence[str] = ()
) -> TransportSecuritySettings:
    """Return Host and Origin checks for the HTTP transport: local addresses plus any given.

    The MCP specification requires an HTTP server to validate the ``Origin`` header, so a web page cannot drive a
    server on the user's machine. Requests without an ``Origin`` (non-browser clients) are accepted.

    Parameters
    ----------
    allowed_hosts : sequence of str
        Extra ``Host`` header values to accept, such as ``example.hf.space``; ``name:*`` accepts any port.
    allowed_origins : sequence of str
        Extra ``Origin`` header values to accept, such as ``https://claude.ai``.

    Returns
    -------
    TransportSecuritySettings
        Settings with DNS rebinding protection on.
    """
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[f"{host}:*" for host in LOCAL_HOSTS] + list(LOCAL_HOSTS) + list(allowed_hosts),
        allowed_origins=[f"http://{host}:*" for host in LOCAL_HOSTS] + list(allowed_origins),
    )


def create_http_app(
    client: MagniDataClient,
    path: str = HTTP_PATH,
    allowed_hosts: Sequence[str] = (),
    allowed_origins: Sequence[str] = (),
) -> Starlette:
    """Return the server as a stateless Streamable HTTP ASGI app, for running or mounting in another app.

    Parameters
    ----------
    client : MagniDataClient
        Client for the MagniData API the tools read from.
    path : str
        URL path the MCP endpoint is served at.
    allowed_hosts, allowed_origins : sequence of str
        Accepted beyond local addresses; see `transport_security`.

    Returns
    -------
    Starlette
        The ASGI app. Its lifespan starts the session manager, so a host app that mounts it must run that lifespan.
    """
    return create_server(client).streamable_http_app(
        streamable_http_path=path,
        stateless_http=True,
        json_response=True,
        transport_security=transport_security(allowed_hosts, allowed_origins),
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for ``magnidata-mcp``."""
    parser = argparse.ArgumentParser(
        prog="magnidata-mcp",
        description="Serve MagniData's datasets to MCP clients (Claude, IDEs, agents) as read-only tools.",
        epilog=f"Public demo API: --api-url {DEMO_API_URL}",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get(API_URL_ENV) or DEFAULT_API_URL,
        help=f"MagniData API to read from (default: ${API_URL_ENV}, else {DEFAULT_API_URL}, the Docker stack).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Seconds to wait for one API response (default: %(default)s).",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="stdio for a local MCP client (default), or streamable-http to serve over HTTP.",
    )
    http = parser.add_argument_group("streamable-http options")
    http.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: %(default)s).")
    http.add_argument("--port", type=int, default=8000, help="Port to listen on (default: %(default)s).")
    http.add_argument("--path", default=HTTP_PATH, help="URL path of the MCP endpoint (default: %(default)s).")
    http.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="HOST",
        help="Host header to accept besides localhost; repeat for more. Needed behind a public hostname.",
    )
    http.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="Browser Origin to accept besides localhost; repeat for more.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {PACKAGE_VERSION}")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the MagniData MCP server; see ``magnidata-mcp --help``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    # The HTTP client logs every request at INFO, which would drown the server's own log on stderr.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    try:
        client = MagniDataClient(args.api_url, timeout=args.timeout)
    except ValueError as err:
        parser.error(str(err))
    if args.transport == "stdio":
        create_server(client).run("stdio")
        return
    create_server(client).run(
        "streamable-http",
        host=args.host,
        port=args.port,
        streamable_http_path=args.path,
        stateless_http=True,
        json_response=True,
        transport_security=transport_security(args.allowed_host, args.allowed_origin),
    )
