# Maintaining the Hugging Face Space

`precisionaiinc/magnidata` is a Docker Space for the **CPU Basic** tier (2 vCPUs, no GPU). It runs the API and the
dashboard in one container with two datasets already loaded:

- **COCO128** is built while the image is built, by the app's own demo build from its public download.
- **AgriStress-500** is installed from a bundle in a Hugging Face dataset repo,
  `precisionaiinc/AgriStress-500-Demo`. The originals are 12 GB, which is too much to build inside an image. That
  repo exists only for the Space; it is a downsized derivative, not the dataset, and its name keeps it distinct from
  the real `precisionaiinc/AgriStress-500` (`space-data-card.md` is its card).

The Space repository holds two files from this folder, `README.md` (the Space card and settings) and `Dockerfile`.
The Dockerfile clones MagniData from GitHub at `MAGNIDATA_REF`, so the Space runs whatever that ref contains.
Everything here is an add-on: the compose stack never touches the bundle, the Space settings or demo mode.

## Prepare the AgriStress-500 bundle

Needed once, and again if the feature extraction changes. The bundle is built from a normal AgriStress-500 build, so
the feature table is computed from the full-resolution images.

1. Build it with the app: `docker compose up --build`, then Add dataset, AgriStress-500, Prepare (about 12 GB to
   download and several minutes of feature extraction).
2. Export it. Run this isolated (`-I`), because it opens the downloaded images:

   ```bash
   python -I -m precisionai.magnidata.scripts.export_demo_bundle \
     --data-root precisionai/magnidata --stem agristress500 --name AgriStress-500 \
     --description "451 field images with segmentation masks. Images are downsized for this demo; the features and embeddings come from the full-resolution originals." \
     --out bundles/agristress500
   ```

`--max-size` caps the longest image side (default 2048, about 520 MB for all 451 images). The portal asks for up to
4096 px in its image modal, so `--max-size 4096` (about 1.7 GB) keeps that view fully sharp at the cost of more
decoding on a 2-vCPU box. The exporter renames images to `.jpg`, renames the embeddings' keys with them, and refuses to
finish if any row lacks an image, mask or embedding. Keep `bundles/` out of git.

## Create the data repository and the Space

Needs the `hf` CLI 2.x: `uv tool upgrade huggingface-hub`, or `pip install -U huggingface_hub`. The 1.x series has no
browser login and no `hf spaces` command.

### Log in with access to the organization

```bash
hf auth login
```

It prints `https://hf.co/oauth/device` and a one-time code. Open the page, enter the code, and on the approval page
**grant access to the `precisionaiinc` organization with write permission**, not only your own account. A login that
covers just your account can read the organization but cannot create repositories in it, and `hf repos create` fails with
"Cannot access content at https://huggingface.co/api/repos/create. Make sure your token has the correct permissions."
`hf auth whoami` lists the organization either way, so it does not prove this; the permissions it grants are in
`HfApi().whoami()["auth"]["accessToken"]["fineGrained"]["scoped"]`, where `precisionaiinc` must appear as an entity.

### Create the repositories and upload

```bash
hf repos create precisionaiinc/AgriStress-500-Demo --repo-type dataset
hf upload precisionaiinc/AgriStress-500-Demo deploy/huggingface/space-data-card.md README.md --repo-type dataset
hf upload precisionaiinc/AgriStress-500-Demo bundles/agristress500 agristress500 --repo-type dataset

hf repos create precisionaiinc/magnidata --repo-type space --space-sdk docker --private
```

`hf repos create` makes **public** repositories unless you pass `--private`, and `--repo-type` is required on every
`hf upload`: it defaults to a model repository. (In `hf` 1.x the command was `hf repo create`.) A folder upload that is
interrupted resumes when the same command is run again. The data repo is public on purpose, since AgriStress-500 itself is, and
that means the build needs no token and anyone can rebuild the Space. The Space is created private so it can be tested;
switch it to public in its Settings when it is ready to release.

To keep the data repo private instead, add `--private` to its `hf repos create`, create a fine-grained token with read
access to `precisionaiinc/AgriStress-500-Demo` only, and add it in the Space's Settings as a **secret** named
`HF_TOKEN`. Docker Spaces mount secrets into the build at `/run/secrets/<NAME>`, which is how the Dockerfile reads it.

### Upload the Space's files

The Space builds MagniData from GitHub at the Dockerfile's `MAGNIDATA_REF` (default `main`). To test a branch before it
merges, upload a copy of the Dockerfile that points at the branch; the file in the repository stays as it is:

```bash
mkdir /tmp/space
sed 's|^ARG MAGNIDATA_REF=main$|ARG MAGNIDATA_REF=<branch>|' deploy/huggingface/Dockerfile > /tmp/space/Dockerfile
cp deploy/huggingface/README.md /tmp/space/README.md
hf upload precisionaiinc/magnidata /tmp/space . --repo-type space --commit-message "Build from <branch>"
```

Once the branch has merged, upload the repository's own Dockerfile, which builds `main`, or one with the release tag:

```bash
hf upload precisionaiinc/magnidata deploy/huggingface/Dockerfile Dockerfile --repo-type space
hf upload precisionaiinc/magnidata deploy/huggingface/README.md README.md --repo-type space
```

Do this before the branch is deleted: a Space still pointing at a deleted branch fails its next rebuild.

### Watch the build

```bash
hf spaces info precisionaiinc/magnidata                  # includes the runtime stage: BUILDING, RUNNING, BUILD_ERROR
hf spaces logs precisionaiinc/magnidata --build -f       # the image build; without -f it prints what exists and exits
hf spaces logs precisionaiinc/magnidata -n 100           # the running container's log
hf spaces restart precisionaiinc/magnidata               # restart the container without rebuilding
```

A private Space can be reached at its own address, `https://precisionaiinc-magnidata.hf.space`, by sending
`Authorization: Bearer <token>` with a token that can see it; in a browser, open
`https://huggingface.co/spaces/precisionaiinc/magnidata` while signed in. The first build from a cold cache took about
a minute and a half. On CPU Basic the Space sleeps after 48 hours without a visit (`gcTimeout` 172800 in
`HfApi().get_space_runtime("precisionaiinc/magnidata").raw`) and starts again on the next one.

## Update the Space

To move the Space to a release, set the `MAGNIDATA_REF` default at the top of the `Dockerfile` to the tag and upload
it again (Docker Spaces also pass Space variables to the build as build arguments). To rebuild from the same ref after
new commits land on it, use a factory rebuild in the Space's Settings. A new bundle is a new `hf upload` to the data
repo followed by a rebuild.

## Build and run locally

```bash
docker build -t magnidata-space -f deploy/huggingface/Dockerfile deploy/huggingface \
  --secret id=HF_TOKEN,env=HF_TOKEN          # only for a private data repo
docker run --rm --cpus=2 -p 7860:7860 magnidata-space      # http://localhost:7860
```

This clones `main` from GitHub, so it does not include local uncommitted changes. To try a pushed branch, add
`--build-arg MAGNIDATA_REF=<branch-or-tag>`. Add `--build-arg DEMO_DATA_REPO=` to skip the bundles and get COCO128
only.

## What the build and the container do

1. `source` clones the repository.
2. `dashboard` runs `npm ci && npm run build`.
3. `bundles` downloads the data repo with `huggingface_hub`. A failed download fails the build on purpose.
4. The last stage installs the API requirements and copies the API, `tools/`, `scripts/`, `training/` and the built
   dashboard. It runs `preload_demo.py --bundle ... coco128` as uid 1000: bundles are mounted from their stage, not
   copied, so the image holds one copy of the data, and the build fails here if anything cannot be installed.
5. At run time one Flask process serves `/api` and the dashboard on port 7860.

Settings that matter for the Space, all in the Dockerfile's `ENV`:

- `MAGNIDATA_DEMO_MODE=1`, for a Space open to anyone. The server refuses with 403 every request that would change data
  shared by all visitors or start work: dataset builds (including the demo downloads), creating, merging, re-parenting
  and deleting datasets, notes, partition commits, training, and cancelling, linking or running inference on a run. It
  also keeps the image endpoints' `path` parameters inside the data folder (400 otherwise, which closes an
  arbitrary-file-read on an open server), refuses the unused mask debug endpoint, and caps the 3D view's background
  analyses at 4 running and 50 kept. `GET /api/config` tells the portal to show the matching controls (Add dataset, View
  experiments, the Experiments section, Train, delete, merge, save as dataset, create partitions) disabled with a
  "Disabled for the Demo" tooltip. Unset, nothing changes.
- `MAGNIDATA_STATIC_DIR=/app/dashboard`: the API serves the built dashboard, since there is no nginx.
- `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0` and a 3 s timeout: there is no MLflow server, so a request to it must fail at once.
- `OMP_NUM_THREADS=2` (and the BLAS equivalents): CPU Basic gives 2 vCPUs, but the container can see the host's cores.

Visitors can explore but cannot change anything: the preloaded datasets are part of the image and a restart restores
them. Read-only requests, including the heavier embedding projections, clusters and analyses, stay open because they are
what the demo shows; they are cached and bounded, and the container is small, so a very busy Space is the thing to
watch.
