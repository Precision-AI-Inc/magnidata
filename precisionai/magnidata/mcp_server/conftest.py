# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""A fake MagniData API for the MCP server tests, answering through an in-memory HTTP transport."""

import json
from collections.abc import Callable

import httpx2
import pytest

from precisionai.magnidata.mcp_server.client import MagniDataClient

FAKE_URL = "http://magnidata.test"
SOURCE = "data_user/tiny.csv"
TINY_CSV = (
    "image_path,width,blur,category,note\n"
    "data_user/tiny/a.jpg,640,12.5,1,sharp\n"
    "data_user/tiny/b.jpg,320,3.25,2,\n"
    "data_user/tiny/c.jpg,640,,2,Blurry edge\n"
    "data_user/tiny/d.jpg,1024,40,1,sharp\n"
)
POSITIONS = [[0.123456, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
LABELS = [1, 0, 1, 1]
JPEG = b"\xff\xd8\xff\xe0fake-jpeg"
PNG = b"\x89PNG\r\n\x1a\nfake-png"


class FakeAPI:
    """Answer the read-only MagniData routes for one tiny dataset; record every request."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.routes: dict[str, Callable[[httpx2.Request], httpx2.Response]] = {
            "/api/datasets": lambda _: _json(
                [
                    {
                        "name": "Tiny",
                        "description": "Four images",
                        "source": SOURCE,
                        "row_count": 4,
                        "has_embeddings": True,
                        "parent_sources": [],
                        "deletable": False,
                    }
                ]
            ),
            "/api/datasets/summary": self._summary,
            "/api/dataset/csv": self._csv,
            "/api/embedding/projection": lambda _: _json({"positions": POSITIONS}),
            "/api/embedding/clusters": lambda _: _json({"labels": LABELS}),
            "/api/image": lambda _: httpx2.Response(200, content=JPEG, headers={"content-type": "image/jpeg"}),
            "/api/annotation/mask": lambda _: httpx2.Response(200, content=PNG, headers={"content-type": "image/png"}),
            "/api/annotation/overlay": lambda _: httpx2.Response(
                200, content=JPEG, headers={"content-type": "image/jpeg"}
            ),
        }

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        """Answer one request from `routes`, or 404 like the real API for an unknown path."""
        self.requests.append(request)
        route = self.routes.get(request.url.path)
        if route is None:
            return _json({"error": "not found"}, 404)
        return route(request)

    def last_params(self) -> dict[str, str]:
        """Return the query parameters of the most recent request."""
        return dict(self.requests[-1].url.params)

    @staticmethod
    def _summary(request: httpx2.Request) -> httpx2.Response:
        if request.url.params.get("source") != SOURCE:
            return _json({"error": "dataset not found"}, 404)
        return _json(
            {
                "row_count": 4,
                "column_count": 5,
                "image_column": "image_path",
                "has_embeddings": True,
                "mask_coverage": {"checked": 4, "with_masks": 3},
                "sample_images": ["data_user/tiny/a.jpg"],
            }
        )

    @staticmethod
    def _csv(request: httpx2.Request) -> httpx2.Response:
        if request.url.params.get("source") != SOURCE:
            return _json({"error": "dataset not found"}, 404)
        return httpx2.Response(200, text=TINY_CSV, headers={"content-type": "text/csv"})


def _json(body: object, status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, content=json.dumps(body).encode(), headers={"content-type": "application/json"})


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio, the event loop the MCP SDK's stdio and HTTP servers use."""
    return "asyncio"


@pytest.fixture
def fake_api() -> FakeAPI:
    """Return a fresh fake API; tests may replace entries in its `routes`."""
    return FakeAPI()


@pytest.fixture
def client(fake_api: FakeAPI) -> MagniDataClient:
    """Return a client whose requests the fake API answers in memory."""
    return MagniDataClient(FAKE_URL, transport=httpx2.MockTransport(fake_api))
