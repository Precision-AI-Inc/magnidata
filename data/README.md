# data/

Empty by default — this repo ships with no preloaded sample datasets (see
`precisionai/magnidata/confi.yaml`, which lists none). It's mounted read-only into the API
container by `docker-compose.yml` at `/app/data`.

To register a permanent catalog entry (as opposed to a user-uploaded or demo-built
dataset — see the portal's "Bring Your Own Data" dialog for those), drop a feature CSV
here (and, optionally, a same-stem `.json` embeddings sidecar) and list it in
`../precisionai/magnidata/confi.yaml`. See
[`../precisionai/magnidata/docs/data-contract.md`](../precisionai/magnidata/docs/data-contract.md)
for the exact file formats.

Large CSVs/JSON files added here should go through Git LFS (`.gitattributes` already
declares the filter for `data/*.csv` and `data/*.json`) — run `git lfs install` once per
clone if you haven't already.
