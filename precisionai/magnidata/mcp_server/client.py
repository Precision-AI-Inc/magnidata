# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""A small async client for the read-only part of MagniData's REST API."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlsplit

import httpx2

from precisionai.magnidata.mcp_server.schemas import ImageKind

DEFAULT_API_URL = "http://localhost:5051"
DEMO_API_URL = "https://precisionaiinc-magnidata.hf.space"
DEFAULT_TIMEOUT_SECONDS = 120.0

try:
    PACKAGE_VERSION = version("magnidata-mcp")
except PackageNotFoundError:
    # Running from a source checkout, where only the main magnidata distribution is installed.
    PACKAGE_VERSION = "0.0.0"

_IMAGE_ROUTES: dict[str, str] = {
    "image": "/api/image",
    "mask": "/api/annotation/mask",
    "overlay": "/api/annotation/overlay",
}
# The API lists its precomputed-projections sidecar (<stem>_projections.json) as if it were another model's embeddings.
_NOT_A_VARIANT = "projections"


class MagniDataAPIError(Exception):
    """The MagniData API could not be reached or refused a request; the message says which and why."""


def normalize_api_url(url: str) -> str:
    """Return the API root for `url`: no trailing slash and no trailing ``/api``.

    Parameters
    ----------
    url : str
        Where the MagniData API or dashboard is served, such as ``http://localhost:5051``.

    Returns
    -------
    str
        The URL that ``/api/...`` paths are appended to.

    Raises
    ------
    ValueError
        If `url` is not an absolute http or https URL.
    """
    root = url.strip().rstrip("/")
    root = root.removesuffix("/api")
    parts = urlsplit(root)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"MagniData API URL must be an http:// or https:// URL, got {url!r}")
    return root


class MagniDataClient:
    """Call MagniData's read-only REST endpoints.

    Parameters
    ----------
    api_url : str
        Root of the API, such as ``http://localhost:5051`` (the Docker stack) or `DEMO_API_URL`.
    timeout : float
        Seconds to wait for one response. Projections of a large dataset can take a while on first request.
    transport : httpx2.AsyncBaseTransport, optional
        Replaces the network transport, so tests can answer requests themselves.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_url = normalize_api_url(api_url)
        self._timeout = timeout
        self._transport = transport

    async def list_datasets(self) -> list[dict[str, Any]]:
        """Return the dataset records from ``GET /api/datasets``."""
        datasets = await self._get_json("/api/datasets")
        if not isinstance(datasets, list):
            raise MagniDataAPIError("GET /api/datasets did not return a list")
        return datasets

    async def dataset_summary(self, source: str) -> dict[str, Any]:
        """Return row count, numeric distributions, sample images and mask coverage of one dataset."""
        return await self._get_json("/api/datasets/summary", {"source": source})

    async def dataset_csv(self, source: str) -> str:
        """Return a dataset's feature table as CSV text."""
        response = await self._get("/api/dataset/csv", {"source": source})
        return response.text

    async def embedding_variants(self, source: str) -> list[str]:
        """Return the names of a dataset's comparison embedding models."""
        variants = await self._get_json("/api/dataset/embeddings/variants", {"source": source})
        return [v["variant"] for v in variants if v.get("variant") and v["variant"] != _NOT_A_VARIANT]

    async def embedding_projection(self, source: str, method: str, variant: str | None = None) -> list[list[float]]:
        """Return one ``[x, y, z]`` per table row from ``GET /api/embedding/projection``."""
        params = {"source": source, "method": method} | ({"variant": variant} if variant else {})
        return (await self._get_json("/api/embedding/projection", params))["positions"]

    async def embedding_clusters(self, source: str, k: int, variant: str | None = None) -> list[int]:
        """Return one cluster label per table row from ``GET /api/embedding/clusters``."""
        params = {"source": source, "k": str(k)} | ({"variant": variant} if variant else {})
        return (await self._get_json("/api/embedding/clusters", params))["labels"]

    async def image(self, path: str, kind: ImageKind, max_size: int, alpha: float) -> tuple[bytes, str]:
        """Return an image, its segmentation mask or the two composited, with the response's media type.

        Parameters
        ----------
        path : str
            The image path, as found in the dataset's image column.
        kind : {"image", "mask", "overlay"}
            What to render. Masks and overlays come from the dataset's own annotations.
        max_size : int
            Longest side in pixels.
        alpha : float
            Mask opacity for an overlay, 0 to 1.

        Returns
        -------
        tuple of (bytes, str)
            The encoded image and its media type, such as ``image/jpeg``.
        """
        params = {"path": path, "max_size": str(max_size)}
        if kind == "overlay":
            params["alpha"] = str(alpha)
        response = await self._get(_IMAGE_ROUTES[kind], params)
        media_type = response.headers.get("content-type", "").split(";")[0].strip()
        if not media_type.startswith("image/"):
            raise MagniDataAPIError(
                f"GET {_IMAGE_ROUTES[kind]} returned {media_type or 'no content type'}, not an image"
            )
        return response.content, media_type

    async def _get_json(self, path: str, params: dict[str, str] | None = None) -> Any:
        response = await self._get(path, params)
        try:
            return response.json()
        except ValueError:
            raise MagniDataAPIError(
                f"GET {path} at {self.api_url} did not return JSON; check that the URL points at a MagniData API"
            ) from None

    async def _get(self, path: str, params: dict[str, str] | None = None) -> httpx2.Response:
        headers = {"User-Agent": f"magnidata-mcp/{PACKAGE_VERSION}"}
        try:
            async with httpx2.AsyncClient(
                base_url=self.api_url, timeout=self._timeout, transport=self._transport, headers=headers
            ) as http:
                response = await http.get(path, params=params)
        except httpx2.TimeoutException:
            raise MagniDataAPIError(
                f"The MagniData API at {self.api_url} did not answer GET {path} within {self._timeout:g} s"
            ) from None
        except httpx2.TransportError as err:
            raise MagniDataAPIError(
                f"Cannot reach the MagniData API at {self.api_url} ({err}). Start the local stack with "
                f"`docker compose up`, or set MAGNIDATA_API_URL to the public demo, {DEMO_API_URL}."
            ) from None
        if response.status_code >= 400:
            raise MagniDataAPIError(f"GET {path} failed with HTTP {response.status_code}: {_error_text(response)}")
        return response


def _error_text(response: httpx2.Response) -> str:
    """Return the API's ``{"error": ...}`` message, or the start of the body when there is none."""
    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        # An HTML error page (Flask's default 500, a proxy's 502) says nothing an agent can use.
        return response.reason_phrase if not text or text.startswith("<") else text[:200]
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])
    return str(body)[:200]
