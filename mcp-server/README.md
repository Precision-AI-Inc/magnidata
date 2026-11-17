# MagniData MCP server

<!-- mcp-name: io.github.Precision-AI-Inc/magnidata -->

`magnidata-mcp` lets Claude, ChatGPT-style agents, IDE assistants and any other
[Model Context Protocol](https://modelcontextprotocol.io) client explore image datasets in
[MagniData](https://github.com/Precision-AI-Inc/magnidata). An agent can list datasets, read per-image features
such as blur, exposure, colour and annotation coverage, filter and sort them, get 3D embedding projections and
clusters, and look at images, segmentation masks and overlays.

Every tool is read-only. The server is a thin client of a running MagniData API. It never calls an endpoint that
changes data, builds a dataset or trains a model, and it holds no data of its own.

## Tools

| Tool | What it returns | MagniData endpoint |
|---|---|---|
| `list_datasets` | Each dataset's name, `source` id, row count and whether it has embeddings | `GET /api/datasets` |
| `describe_dataset` | Every column with type and statistics, the image column, mask coverage, sample image paths | `GET /api/datasets/summary`, `GET /api/dataset/csv` |
| `query_rows` | A page of the per-image feature table, filtered (`==`, `!=`, `<`, `<=`, `>`, `>=`, `contains`, `empty`, `not_empty`) and sorted | `GET /api/dataset/csv` |
| `embedding_projection` | 3D PCA, t-SNE or LLE coordinates, one point per row, optionally with table columns attached | `GET /api/embedding/projection` |
| `embedding_clusters` | Spherical k-means clusters: sizes, example images and per-row labels | `GET /api/embedding/clusters` |
| `get_image` | An image, its segmentation mask or the mask drawn over it, as image content (512 px by default) | `GET /api/image`, `/api/annotation/mask`, `/api/annotation/overlay` |

Each tool has a title and the `readOnlyHint` annotation. Rows, projection points and cluster labels share one
`row_index`, so an agent can join them: for example, find the blurriest images with `query_rows`, see where they
fall in `embedding_projection`, then look at a few with `get_image`.

## Which MagniData it talks to

By default the server reads from `http://localhost:5051`, the API of the local Docker stack (`docker compose up`
in the repository root; see the [main README](https://github.com/Precision-AI-Inc/magnidata#readme)).

To try it without running anything locally, point it at the public demo on Hugging Face, which has COCO128 and
AgriStress-500 loaded:

```text
https://precisionaiinc-magnidata.hf.space
```

Set the address with `--api-url` or the `MAGNIDATA_API_URL` environment variable. The demo is a shared,
read-only deployment on free hardware: it may take a minute to wake up after a quiet period.

## Install

The server needs Python 3.10 or newer. The examples use [`uvx`](https://docs.astral.sh/uv/), which downloads and
runs the package in its own environment.

```bash
uvx magnidata-mcp --help
```

Until `magnidata-mcp` is on PyPI, run it straight from GitHub instead. Everywhere below, replace
`uvx magnidata-mcp` with:

```bash
uvx --from "git+https://github.com/Precision-AI-Inc/magnidata#subdirectory=mcp-server" magnidata-mcp
```

With pip, `pip install magnidata-mcp` (or `pip install "git+https://github.com/Precision-AI-Inc/magnidata#subdirectory=mcp-server"`)
installs a `magnidata-mcp` command.

## Configure your client

### Claude Code

```bash
# Local Docker stack
claude mcp add magnidata -- uvx magnidata-mcp

# Public demo
claude mcp add magnidata-demo -- uvx magnidata-mcp --api-url https://precisionaiinc-magnidata.hf.space
```

Add `--scope user` to make it available in every project. Then ask, for example, "Using MagniData, which COCO128
images are the blurriest, and do they cluster together?"

### Claude Desktop

Open **Settings → Developer → Edit Config** (the file is `claude_desktop_config.json`: in
`~/Library/Application Support/Claude/` on macOS, `%APPDATA%\Claude\` on Windows), add the server and restart
Claude Desktop:

```json
{
  "mcpServers": {
    "magnidata": {
      "command": "uvx",
      "args": ["magnidata-mcp"],
      "env": {
        "MAGNIDATA_API_URL": "https://precisionaiinc-magnidata.hf.space"
      }
    }
  }
}
```

Leave out `env` to use the local stack. If Claude Desktop reports that it cannot find `uvx`, use the full path
that `which uvx` (or `where uvx` on Windows) prints as `command`.

### Other MCP clients

Any client that launches stdio servers takes the same command, arguments and environment. Most use the
`mcpServers` shape above (Cursor's `.cursor/mcp.json`, Windsurf, Cline). VS Code's `.vscode/mcp.json` uses a
`servers` key:

```json
{
  "servers": {
    "magnidata": {
      "type": "stdio",
      "command": "uvx",
      "args": ["magnidata-mcp", "--api-url", "https://precisionaiinc-magnidata.hf.space"]
    }
  }
}
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--api-url URL` | `$MAGNIDATA_API_URL`, else `http://localhost:5051` | MagniData API to read from. A trailing `/api` is accepted. |
| `--timeout SECONDS` | `120` | Longest wait for one API response. A first t-SNE or LLE projection of a large dataset can be slow. |
| `--transport` | `stdio` | `stdio` for a local client, or `streamable-http` to serve over HTTP (below). |
| `--host`, `--port`, `--path` | `127.0.0.1`, `8000`, `/mcp` | Where the HTTP transport listens. |
| `--allowed-host HOST` | localhost only | `Host` header to accept over HTTP; repeat for more. Needed behind a public hostname. |
| `--allowed-origin ORIGIN` | localhost only | Browser `Origin` to accept over HTTP; repeat for more. |

## Streamable HTTP

```bash
magnidata-mcp --transport streamable-http --port 8000
# MCP endpoint: http://127.0.0.1:8000/mcp
```

This mode serves the same read-only tools over [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports#streamable-http),
stateless and with JSON responses, so any number of clients can share one process. It is off unless you start it.
It checks every request's `Host` and `Origin` headers, as the MCP specification requires: only local addresses
are accepted unless you add others with `--allowed-host` and `--allowed-origin`. Requests without an `Origin`
header, which is how non-browser clients call, are accepted.

To serve the endpoint from another ASGI application, such as next to the MagniData API on the Hugging Face
Space, build the app with `create_http_app` and mount it. The host app must run the mounted app's lifespan, which
starts the MCP session manager:

```python
from precisionai.magnidata.mcp_server import MagniDataClient, create_http_app

mcp_app = create_http_app(
    MagniDataClient("http://127.0.0.1:7860"),
    allowed_hosts=["precisionaiinc-magnidata.hf.space"],
)
```

Nothing deploys this endpoint today. Before exposing it publicly, add rate limiting in front of it; the server does
no rate limiting of its own.

## Development

The code lives in the main repository at
[`precisionai/magnidata/mcp_server/`](https://github.com/Precision-AI-Inc/magnidata/tree/main/precisionai/magnidata/mcp_server),
with its tests next to it, and is tested, linted and type-checked with the rest of the repository
(`pip install -e ".[dev]"`, then `pytest precisionai/magnidata/mcp_server`). From a checkout you can also run it as
`python -m precisionai.magnidata.mcp_server`.

This directory only packages it: `hatch_build.py` copies that package, without its tests, into the
`magnidata-mcp` sdist and wheel, so the distribution installs nothing but the MCP SDK and an HTTP client. Build
with `uv build mcp-server` (or `python -m build mcp-server`) from the repository root. The version comes from the
repository's `vX.Y.Z` tag, like `magnidata`'s own.

## Publishing

The PyPI package and the [MCP Registry](https://registry.modelcontextprotocol.io) entry are published together by
the **Release MCP server** workflow (`.github/workflows/release-mcp.yml`). It builds and tests the package, checks
that its version matches the tag, publishes it to PyPI with trusted publishing, stamps that version into
`server.json` and publishes `server.json` to the registry as `io.github.Precision-AI-Inc/magnidata`. The `version`
fields in the committed `server.json` are placeholders that the workflow overwrites.

One-time setup, before the first run:

1. **PyPI.** Whoever owns the PyPI account or organization that will own `magnidata-mcp` adds a trusted publisher
   for a new project ("pending publisher"): project `magnidata-mcp`, owner `Precision-AI-Inc`, repository
   `magnidata`, workflow `release-mcp.yml`, environment `pypi`.
2. **GitHub.** A repository admin creates the `pypi` environment under **Settings → Environments** if it does not
   exist yet, ideally limited to `v*` tags.
3. **MCP Registry.** Nothing to create and no secret to store: the registry grants a workflow in this repository
   the `io.github.Precision-AI-Inc/*` namespace from its GitHub OIDC token. (The interactive
   `mcp-publisher login github` cannot publish to an organization namespace today, which is why publishing runs in
   GitHub Actions.)

Then, for each release, the one step is to run the workflow on the release tag:

```bash
gh workflow run release-mcp.yml --ref v0.2.0
```
