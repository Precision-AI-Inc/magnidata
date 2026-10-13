import type { ComponentType } from 'react'
import { Architecture } from './Architecture'
import { Cmd, Expect, Goal, Note, Output, Shot, Steps, Term } from './blocks'

export interface Chapter {
  id: string
  title: string
  summary: string
  minutes: number
  Body: ComponentType
}

function Overview() {
  return (
    <>
      <Goal>Understand what each piece does and how data moves between them, before you touch anything.</Goal>
      <p>
        MagniData is a workspace for improving segmentation training data. You bring images (and masks), look
        at them from many angles, train a model, and use what the model finds hard to decide which images to
        add, remove or look at again. Everything below runs on your own machine.
      </p>
      <Architecture />
      <h3>The pieces</h3>
      <ul>
        <li><strong>web (port 5175)</strong> serves this portal. It forwards <code>/api</code> calls to the API.</li>
        <li><strong>api (port 5051)</strong> reads and writes your datasets, serves images, masks and overlays,
          computes projections and clusters, and lets the portal browse training runs.</li>
        <li><strong>mlflow (port 5052)</strong> stores every training run: parameters, per-epoch metrics, the
          model checkpoint and the masks the model predicted. Its own UI is a bonus, not a requirement.</li>
        <li><strong>trainer</strong> waits for runs scheduled from the portal and trains them. The API never
          trains anything itself.</li>
        <li><strong>data folders</strong>: dataset CSVs and images live under <code>precisionai/magnidata/data_user/</code>;
          folders of images you want to turn into datasets live in <code>image_sets/</code>.</li>
      </ul>
      <h3>The loop you will run in this guide</h3>
      <Steps>
        <li>Load <strong>AgriStress-500</strong>, a set of field photographs with segmentation masks.</li>
        <li>From the command line, <Term>partition</Term> it into smaller tiled sub-datasets. They appear in the
          <Term>lineage graph</Term> as children of AgriStress-500.</li>
        <li>In the portal, inspect a sub-dataset, explore it in the Visual Explorer, and preview images with their
          ground-truth masks.</li>
        <li>Train a model on it and read the results in Experiments.</li>
        <li>Find the images the model finds persistently hard, save them as a new dataset, train again, and compare.</li>
      </Steps>
      <h3>Words used in this guide</h3>
      <dl className="guide-glossary">
        <dt>Dataset</dt><dd>A CSV with one row per image (its path plus feature columns), the images, and optionally masks.</dd>
        <dt>Lineage</dt><dd>Which dataset another was derived from. Shown as arrows in the workspace graph.</dd>
        <dt>Partition</dt><dd>Cutting images into fixed-size tiles and dividing them into sub-datasets.</dd>
        <dt>Run / experiment</dt><dd>One training job and its recorded results. Experiments group runs.</dd>
        <dt>Ground truth</dt><dd>The mask that ships with the dataset. What the model should learn to reproduce.</dd>
        <dt>Prediction</dt><dd>The mask the trained model produced. Never the same file as the ground truth.</dd>
        <dt>Hardness</dt><dd>How persistently large an image&apos;s training loss is across epochs. High means the model keeps struggling.</dd>
        <dt>Complement</dt><dd>Extra images from a larger pool that resemble the ones you selected.</dd>
      </dl>
    </>
  )
}

function Install() {
  return (
    <>
      <Goal>Start the full stack and confirm it is healthy. About five minutes, most of it the first build.</Goal>
      <h3>You need</h3>
      <ul>
        <li>Git and Docker with Compose v2 (<code>docker compose version</code> prints a version).</li>
        <li>About 20 GB of free disk for the images this guide uses, and a few GB for the Docker images.</li>
        <li>Optional: an NVIDIA GPU with the NVIDIA Container Toolkit to train faster. CPU works, just slower.</li>
      </ul>
      <Steps>
        <li>Get the code.<Cmd id="clone" /></li>
        <li>Build and start everything. Leave this terminal open; it shows the logs.<Cmd id="up" /></li>
        <li>In another terminal, check that the four services are up.<Cmd id="ps" /></li>
        <li>Open <a href="http://localhost:5175">http://localhost:5175</a> in your browser.</li>
      </Steps>
      <Expect>
        <p><code>docker compose ps</code> lists <code>web</code>, <code>api</code>, <code>mlflow</code> and <code>trainer</code> as
          running. The portal opens on the <strong>Workspace</strong> page with five summary cards (Datasets, Samples,
          Explorer ready, Runs, Trainer). Trainer says <strong>Ready</strong> once the worker has reported in.</p>
      </Expect>
      <Shot name="workspace-source" />
      <Note tone="warn" title="502 Bad Gateway or &quot;Dataset could not be restored&quot;">
        The portal starts a few seconds before the API. If you load the page in that window you see a 502.
        Wait for <code>api</code> to print <em>Running on</em> in the logs, then press <strong>Retry</strong> or reload.
        You can also check it directly:<Cmd id="health" />
      </Note>
      <h3>Train on a GPU (optional)</h3>
      <p>Tell Compose to also load the GPU file, restart, and check that the trainer sees the GPU:</p>
      <Cmd id="gpuEnv" />
      <Cmd id="up" />
      <Cmd id="gpuCheck" />
      <p>The last command should print <code>True</code>. Run details show a GPU or CPU badge and an ETA.</p>
      <h3>Stop, restart and reset</h3>
      <Cmd id="stop" note="Stop everything. Your datasets and runs are kept." />
      <Cmd id="reset" note="Also delete the stored runs and notes (the MLflow and database volumes). Datasets on disk stay." />
      <Cmd id="logs" note="Follow the API log when something looks wrong." />
    </>
  )
}

function AgriStress() {
  return (
    <>
      <Goal>Get the AgriStress-500 dataset into the workspace using the one-click demo.</Goal>
      <p>
        AgriStress-500 is a public set of field photographs with colour-coded segmentation masks, published by
        Precision AI on Hugging Face (<code>precisionaiinc/AgriStress-500</code>). The demo downloads the images
        and masks, computes the feature columns and registers the dataset for you. Check the dataset card on
        Hugging Face for its terms of use before you redistribute it.
      </p>
      <Steps>
        <li>On the <strong>Workspace</strong> page click <strong>Add dataset</strong>.</li>
        <li>Choose <strong>Load Demo</strong>, then <strong>AgriStress-500</strong>.</li>
        <li>Watch the progress badge in the page header. The download is large (roughly 12 GB of PNG images and masks),
          so expect this to take a while on a normal connection. You can keep using the portal meanwhile.</li>
      </Steps>
      <Shot name="byod-demo" />
      <Expect>
        <p>When the build finishes, <strong>AgriStress-500</strong> appears as a card in the lineage graph with its
          sample count (451 images). The <strong>Datasets</strong> and <strong>Samples</strong> cards at the top update.</p>
      </Expect>
      <Note tone="tip" title="Using your own images instead">
        Put a folder with an <code>images/</code> subfolder (and optionally <code>masks/</code> or
        <code> labels/</code>) under <code>image_sets/</code>, then use <strong>Add dataset → Prepare From Server Folder</strong>.
        There is no size limit because the images are read in place. <strong>Upload</strong> works too for smaller sets.
        The rest of the guide is identical. Chapter 9 covers this in detail.
      </Note>
    </>
  )
}

function Partition() {
  return (
    <>
      <Goal>Cut AgriStress-500 into small tiles and split them into sub-datasets that show up in the lineage graph.</Goal>
      <p>
        Full-size field images are too large to train on directly. The partition tool crops every image (and its
        mask) into 256 × 256 tiles, divides the tiles into several sub-datasets, builds each one with the same
        pipeline as <em>Prepare From Server Folder</em>, and registers it as a child of AgriStress-500. This is the
        only step in the guide that uses the command line.
      </p>
      <h3>Before you start</h3>
      <p>Run these from the repository root, with the stack from chapter 2 still running. You need Python 3.10+:</p>
      <Cmd id="venv" />
      <Steps>
        <li>Preview what it will do. Nothing is written.<Cmd id="partitionDry" /></li>
        <li>Run it for real. It tiles the images, then builds and registers each part (about a minute per part for this size).<Cmd id="partition" /></li>
      </Steps>
      <Output>{`Source data_user/agristress500.csv: 8 images (8 with masks) -> 1771 tiles of 256x256, split by image into 2 parts
  part 1: 4 images, 909 tiles
  part 2: 4 images, 862 tiles
  part 1: writing 909 tiles to image_sets/agristress-500_tiles-part1
  part 2: writing 862 tiles to image_sets/agristress-500_tiles-part2
  AgriStress-500 tiles part 1: extracting_features 30% Extracting features…
  AgriStress-500 tiles part 1: done 100% Done
  AgriStress-500 tiles part 2: extracting_features 30% Extracting features…
  AgriStress-500 tiles part 2: done 100% Done
Done. Open http://localhost:5175 to see the sub-datasets under data_user/agristress500.csv in the lineage graph.`}</Output>
      <Expect>
        <p>Reload the Workspace page. Two new cards, <strong>AgriStress-500 tiles part 1</strong> and <strong>part 2</strong>, sit to
          the right of AgriStress-500 with an arrow from it to each. Their experiment badges are gray: no runs yet.</p>
      </Expect>
      <Shot name="workspace-lineage" callouts={[
        { x: 22, y: 50, text: 'The source dataset' },
        { x: 70, y: 36, text: 'A tiled sub-dataset (part 1)' },
        { x: 70, y: 70, text: 'A tiled sub-dataset (part 2)' },
      ]} />
      <h3>What it did</h3>
      <ul>
        <li><strong>Sampled</strong> 8 images (<code>--limit-images 8</code>) so the guide finishes quickly. Leave the flag out to use every image (451 in the full dataset).</li>
        <li><strong>Tiled</strong> each image and its mask into full 256 × 256 crops (partial edge tiles are dropped).</li>
        <li><strong>Split by image</strong>: every tile of one source image lands in the same part, so one part can safely
          validate a model trained on another. Use <code>--split-by tile</code> to spread individual tiles instead.</li>
        <li><strong>Wrote</strong> the tiles to <code>image_sets/&lt;name&gt;-partN/images</code> and <code>masks</code>, with a <code>tile_manifest.csv</code>.</li>
        <li><strong>Built and registered</strong> each part and linked it under the source through the API. The tiles have no embeddings yet; chapter 10 adds them.</li>
      </ul>
      <h3>Options</h3>
      <table className="guide-table">
        <thead><tr><th>Flag</th><th>Default</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><code>--source</code></td><td><code>data_user/agristress500.csv</code></td><td>Dataset to partition</td></tr>
          <tr><td><code>--parts</code></td><td>2</td><td>Number of sub-datasets</td></tr>
          <tr><td><code>--tile</code></td><td>256</td><td>Tile size in pixels</td></tr>
          <tr><td><code>--split-by</code></td><td><code>image</code></td><td><code>image</code> or <code>tile</code></td></tr>
          <tr><td><code>--limit-images</code></td><td>all</td><td>Use a seeded sample of N source images</td></tr>
          <tr><td><code>--seed</code></td><td>0</td><td>Seed for sampling and splitting</td></tr>
          <tr><td><code>--name</code></td><td><code>AgriStress-500 tiles</code></td><td>Name prefix; parts are called &quot;&lt;name&gt; part N&quot;</td></tr>
          <tr><td><code>--dry-run</code></td><td>off</td><td>Print the plan and exit</td></tr>
          <tr><td><code>--no-register</code></td><td>off</td><td>Only write the tiles</td></tr>
          <tr><td><code>--data-root</code></td><td><code>precisionai/magnidata</code></td><td>Host folder that holds <code>data_user/</code> (what the API sees as its data root)</td></tr>
          <tr><td><code>--image-sets</code></td><td><code>image_sets</code></td><td>Host folder the tiles are written to (mounted into the API as <code>image_sets/</code>)</td></tr>
          <tr><td><code>--api</code>, <code>--portal</code></td><td>localhost:5051 / 5175</td><td>Where the stack is listening</td></tr>
        </tbody>
      </table>
      <Note tone="tip" title="Safe to repeat">
        Running the same command again skips parts that are already tiled and registered, so you can resume after
        an interruption. Use <code>--help</code> to see every option:<Cmd id="partitionHelp" />
      </Note>
      <Note title="Going bigger">
        The whole dataset is about 100,000 tiles and takes roughly a minute of feature extraction per 1,000 tiles.
        When you are ready: <Cmd id="partitionFull" />
      </Note>
    </>
  )
}

function Workspace() {
  return (
    <>
      <Goal>Learn the Workspace page: summary, lineage graph, detail panel, merging and badges.</Goal>
      <h3>The lineage graph</h3>
      <ul>
        <li><strong>Click</strong> a dataset to select it. Its ancestors and descendants stay bright and the rest fade, so you see where it came from and what was derived from it.</li>
        <li><strong>Double-click</strong> (or press Enter) to open it in the Visual Explorer.</li>
        <li><strong>Drag</strong> to pan, <strong>scroll</strong> to zoom, and use the minimap in the corner for large graphs.</li>
        <li>Arrow keys move between connected datasets; <strong>Esc</strong> clears the selection.</li>
      </ul>
      <h3>The detail panel</h3>
      <p>Select <strong>AgriStress-500 tiles part 1</strong>. A panel slides in with the dataset&apos;s name, three actions and five tabs.</p>
      <Steps>
        <li><strong>Open in Visual Explorer</strong> loads it for exploration (next chapter).</li>
        <li><strong>View experiments</strong> opens Experiments filtered to this dataset (chapter 7).</li>
        <li>The trash icon deletes a dataset you created (not built-in ones), after a confirmation.</li>
      </Steps>
      <Shot name="panel-statistics" />
      <Shot name="panel-images" />
      <Shot name="panel-annotations" />
      <h3>Merge datasets</h3>
      <p>
        Hold <strong>Shift</strong> (or Ctrl/Cmd) and click two or more datasets. The panel changes to <strong>Merge datasets</strong>.
        Name the result and press <strong>Merge datasets</strong>. The new dataset holds the union of the images
        (each image once), is linked under every source, and starts with no experiments.
      </p>
      <Shot name="merge" />
      <Note tone="warn" title="Same columns required">
        Merging needs the datasets to share the same CSV columns. Datasets built with the same pipeline do; if you
        merge datasets from different tools you will get a clear error instead of a broken file.
      </Note>
      <h3>Experiment badges</h3>
      <p>
        Every card has a small flask. <strong>Yellow and pulsing</strong> means a run on that dataset is queued or training.
        <strong> Gray</strong> means none; hover it for the counts (for example &quot;3 experiments, none running&quot;).
      </p>
    </>
  )
}

function Explorer() {
  return (
    <>
      <Goal>Explore a dataset visually, look at individual tiles with their masks, and save a selection as a new dataset.</Goal>
      <p>Select <strong>AgriStress-500 tiles part 1</strong> in the Workspace and press <strong>Open in Visual Explorer</strong>.</p>
      <Shot name="explorer" callouts={[
        { x: 38, y: 24, text: 'Parallel coordinates: drag on any axis to keep only a range of values' },
        { x: 25, y: 70, text: 'Scatter view of every tile; colour it by any column' },
        { x: 62, y: 70, text: 'Results table: only the rows that pass your filters' },
        { x: 90, y: 50, text: 'Explorer setup: train, export, save, complement, overlays' },
      ]} />
      <h3>Filter and look</h3>
      <Steps>
        <li>Drag on an axis in the <strong>parallel coordinates</strong> to keep only tiles in that range. Several axes combine.
          The table, the scatter view and the counts in the header all follow.</li>
        <li>Use <strong>Color by</strong> in <em>Explorer setup</em> to colour the scatter by a feature such as <code>blur_laplacian</code> or <code>complexity_score</code>.</li>
        <li>The left navigation lists the other views of the same dataset: Charts, Schema, Table and Overview.</li>
      </Steps>
      <Note title="Embeddings">
        The 3D embeddings view needs a pre-computed embeddings file. The tiled sub-datasets have none, so the
        Embeddings source is greyed out for them. The feature-based views work without it. Chapter 10 shows how to add embeddings.
      </Note>
      <h3>Preview a tile: image, mask, overlay, prediction</h3>
      <p>Click a point in the scatter or a row in the table to open the preview. Its tabs show different things:</p>
      <Shot name="preview-image" />
      <Shot name="preview-overlay" />
      <ul>
        <li><strong>Image</strong>: the photograph only.</li>
        <li><strong>Mask</strong> and <strong>Overlay</strong>: the dataset&apos;s <em>ground-truth</em> annotation, alone or drawn over the image. A caption says &quot;Ground truth · dataset annotation&quot;.</li>
        <li><strong>Prediction</strong> (appears once you overlay a run, chapter 7): what the trained model produced for the same tile, captioned with its epoch.</li>
      </ul>
      <Shot name="preview-prediction" />
      <Note tone="warn" title="Ground truth and prediction are never mixed">
        If a run has no saved prediction for a tile, the Prediction tab says so and offers <strong>Run inference</strong>.
        It never falls back to showing the ground truth, so what you see under Prediction is always the model&apos;s output.
      </Note>
      <h3>Turn what you see into a dataset</h3>
      <p>
        Filter to the tiles you care about, then in <em>Explorer setup → Workflow</em> type a name under
        <strong> Save selection as a dataset</strong> and press <strong>Save N samples as dataset</strong>. A message offers
        <strong> View in workspace</strong>: the new dataset is a child of the one you were exploring. (<strong>Export selection</strong>
        still downloads a CSV if that is what you need.) You can also stage <strong>Remove image</strong> in the preview, or
        use <strong>Complement</strong> to add similar images from the Cloud dataset.
      </p>
      <Shot name="save-selection" />
    </>
  )
}

function Train() {
  return (
    <>
      <Goal>Train a segmentation model on a sub-dataset and read the results.</Goal>
      <Steps>
        <li>In the Visual Explorer press <strong>Train model</strong> (Explorer setup → Workflow).</li>
        <li>Pick a <strong>preset</strong>: <strong>Tiny UNet</strong> is small and trains fastest; <strong>LR-ASPP MobileNetV3</strong> is larger and
          starts from ImageNet-pretrained weights (downloaded on first use). Give the run a name and, for a quick first run, set <em>Epochs</em> to 5.
          The other settings (batch size, image size, learning rate, validation fraction, how many hardest predictions to save) have sensible defaults.</li>
        <li>Press <strong>Start training</strong>. The run is scheduled; the trainer picks it up and starts.</li>
      </Steps>
      <Shot name="train-dialog" />
      <Expect>
        <p>The portal takes you to <strong>Experiments</strong> and the header shows the run&apos;s status and progress (preparing data, indexing masks,
          training epochs, validating, scoring). On the Workspace page the dataset&apos;s flask badge turns yellow while it runs.
          A small run like this one finishes in about a minute on a GPU, a few minutes on CPU.</p>
      </Expect>
      <Shot name="run-live" />
      <h3>Read the results</h3>
      <p>Open <strong>Experiments</strong> in the left navigation.</p>
      <Shot name="experiments" />
      <ul>
        <li>Each row is a run: status, preset, dataset, duration, device and headline metrics (<code>val_loss</code>, <code>val_miou</code>).</li>
        <li>Click a run for its <strong>curves, parameters, tags and artifacts</strong>. Use <strong>Cancel</strong> to stop a running one.</li>
        <li>Tick several runs and press <strong>Compare</strong> to overlay their curves and metrics side by side.</li>
        <li>The status chips and search box narrow the table; <strong>Columns</strong> chooses which metrics and parameters show.</li>
      </ul>
      <Shot name="run-detail" />
      <h3>See the run on your images</h3>
      <p>
        Back in the Visual Explorer, choose the run in <strong>Experiment overlay → Run</strong>. The dataset gains per-image
        loss columns (<code>loss_total</code>, its mean, trend and rank, plus <code>hardness</code>) and a <code>split</code> column
        (train or validation). Use the <strong>epoch scrubber</strong> to see how each image&apos;s loss changed over training.
        The preview now has a <strong>Prediction</strong> tab.
      </p>
      <Shot name="run-overlay" />
    </>
  )
}

function Loop() {
  return (
    <>
      <Goal>Use what the model finds hard to build a better dataset, train on it, and check that it helped.</Goal>
      <Steps>
        <li>With the run overlaid, open <em>Explorer setup → Appearance → <strong>Choose visible fields</strong></em>. Under <strong>Training</strong>
          press <strong>High-loss samples</strong> to brush the images that stay hard across epochs. The counts in the header and the
          <strong> Save … samples as dataset</strong> button now refer to just those tiles. Open a few in the preview and compare the ground truth with the prediction.</li>
        <li>Name the set under <strong>Save selection as a dataset</strong> (for example <em>Hard tiles</em>) and save it, then press
          <strong> View in workspace</strong>. It appears in the lineage graph under part 1.</li>
        <li>Select the new dataset, press <strong>Open in Visual Explorer</strong>, then <strong>Train model</strong>. While it trains its flask is yellow.</li>
        <li>When it finishes, press <strong>View experiments</strong> on the dataset to see only its runs. To compare, open Experiments, tick two
          runs and press <strong>Compare</strong>.</li>
        <li>Optional: merge the hard tiles with the other part to rebalance, or use <strong>Complement</strong> to pull in similar tiles from a larger pool.</li>
      </Steps>
      <Shot name="workspace-badges" />
      <Shot name="experiments-filtered" />
      <Expect>
        <p>The graph shows your new dataset under its parent, with a yellow flask while it trains and gray afterwards.
          <strong> View experiments</strong> on that dataset lists only its runs. Comparing shows per-image loss change from the first run to the second.</p>
      </Expect>
      <Note tone="tip" title="Read results with care">
        Compare runs on the same validation images. Partitioning by image (the default) keeps tiles of one photograph
        in one part, so a part you did not train on is a fair check.
      </Note>
    </>
  )
}

function OwnData() {
  return (
    <>
      <Goal>Turn a folder of your own images (and masks, if you have them) into a dataset in the workspace.</Goal>
      <p>
        You have three ways in. The first is the one to use for almost everything: the images stay where you put them,
        there is no size limit, and it is the only route that supports masks (which training needs).
      </p>
      <table className="guide-table">
        <thead><tr><th>Route</th><th>Use it when</th><th>Limits</th></tr></thead>
        <tbody>
          <tr><td><strong>A. Server folder</strong></td><td>You have images on disk, with or without masks or annotations. <em>Recommended.</em></td><td>None. Read in place; no copy, no upload.</td></tr>
          <tr><td><strong>B. Upload in the browser</strong></td><td>A small set, and you want to attach COCO annotations or an embeddings file as you create it.</td><td>500 images, 16 GiB in total, embeddings file up to 100 MB. No masks.</td></tr>
          <tr><td><strong>C. An existing CSV</strong></td><td>You already have a dataset CSV (for example from the feature tool) and just want to look at it, or to register it permanently.</td><td>The images must be readable by the API.</td></tr>
        </tbody>
      </table>

      <h3>A. From a folder on your disk</h3>
      <p>Arrange your files like this. Only <code>images/</code> is required:</p>
      <Output>{`my-field-survey/
  images/            required: .png .jpg .jpeg .webp .bmp .tif .tiff
    plot-a/          optional subfolders: the folder name becomes each image's cluster
      IMG_0001.jpg
    plot-b/
      IMG_0002.jpg
  masks/             optional: IMG_0001.png ... colour-coded segmentation masks (black = background)
  labels/            optional: IMG_0001.json ... COCO polygon annotations`}</Output>
      <ul>
        <li><strong>Masks</strong> are PNGs named like the image, one colour per class on a black background. You need them to <em>train</em>; they also drive the Mask and Overlay previews.</li>
        <li><strong>COCO labels</strong> (<code>labels/&lt;name&gt;.json</code>, polygon segmentation) add coverage, instance and colour-similarity columns, and are drawn in the Mask and Overlay previews.</li>
        <li>Use <strong>unique file names</strong> across subfolders: results (and later, embeddings) are matched by file name.</li>
      </ul>
      <Steps>
        <li>Copy the folder into <code>image_sets/</code> at the repository root (create it if it does not exist). The Docker stack mounts it into the API.<Cmd id="copyFolder" /></li>
        <li>In the portal press <strong>Add dataset</strong> and choose <strong>Prepare From Server Folder</strong>.</li>
        <li>Find your folder in the list (it shows how many images and labels it found) and press <strong>Prepare</strong>.</li>
        <li>Watch the progress badge in the header: indexing, then feature extraction. About a minute per 1,000 small images.</li>
      </Steps>
      <Shot name="byod-choose" />
      <Shot name="byod-folder" />
      <Expect>
        <p>A new dataset named after the folder appears as its own card in the lineage graph, with its sample count. Select it and
          open the panel: statistics, a sample of images, and annotation coverage are all there. Everything in chapters 5 to 8 now applies to it.</p>
      </Expect>
      <Shot name="workspace-own" />
      <Note tone="tip" title="Rebuilding a folder">
        A folder that has been built shows <strong>Added</strong> and cannot be prepared again. To rebuild it after changing the files,
        delete the dataset (trash icon in its panel), then <strong>Prepare</strong> again. The images themselves are never modified.
      </Note>

      <h3>B. Upload from the browser</h3>
      <Steps>
        <li><strong>Add dataset → Create New Dataset</strong>.</li>
        <li>Give it a <strong>Name</strong> and choose your <strong>Images</strong> (a whole folder, or files).</li>
        <li>Optional: <strong>Annotations</strong>, COCO JSON files matched to images by file name. Optional: an <strong>Embeddings</strong> file
          (see the next chapter).</li>
        <li>Press <strong>Create Dataset</strong>. The upload runs in the background; the header shows its progress.</li>
      </Steps>
      <Shot name="byod-upload" />
      <Note title="Masks cannot be uploaded">
        The upload form takes annotations but not masks. To train on your own data, use route A and include <code>masks/</code>.
      </Note>

      <h3>C. You already have a CSV</h3>
      <ul>
        <li><strong>Just look at it:</strong> press <strong>Upload CSV</strong> in the Workspace toolbar. The file opens in the Visual Explorer for
          this session. It is not added to the lineage graph.</li>
        <li><strong>Register it permanently:</strong> put the CSV in the repository&apos;s <code>data/</code> folder (read-only in the API container),
          add an entry to <code>precisionai/magnidata/confi.yaml</code>, and restart the API. It then appears in the graph like any other dataset.</li>
      </ul>
      <Output>{`datasets:
  - name: My Field Survey
    source: data/my_field_survey.csv
    description: Spring survey, plots A to D`}</Output>
      <Cmd id="restartApi" />
      <p>To compute the CSV from a folder of images yourself, with the same columns the portal produces:</p>
      <Cmd id="features" />
      <h3>Then continue with your data</h3>
      <ul>
        <li>Train on it from the Visual Explorer (it needs masks).</li>
        <li>Tile it like AgriStress-500: <code>partition_dataset --source data_user/&lt;name&gt;.csv</code>, where the file name is the dataset name in lower case with
          anything other than letters, digits, <code>_</code> and <code>-</code> replaced by <code>_</code> (for example <em>My Field Survey</em> becomes <code>my_field_survey.csv</code>).
          Your images must be at least as large as the tile size.</li>
        <li>Add embeddings for the 3D view: next chapter.</li>
      </ul>
    </>
  )
}

function Embeddings() {
  return (
    <>
      <Goal>Add embeddings to a dataset so the Visual Explorer can draw it as a 3D map and cluster it.</Goal>
      <p>
        An <Term>embedding</Term> is a list of numbers that describes what an image looks like, produced by a vision model. Images that look alike get
        similar numbers. With embeddings the Visual Explorer&apos;s <strong>Embeddings</strong> source shows every image as a point in 3D (PCA, t-SNE
        or LLE), colours it by ground-truth cluster or K-means, and finds look-alikes. Without them everything else still works: the parallel
        coordinates, scatter, table and training. They are optional and not computed in the portal.
      </p>
      <Shot name="embeddings-view" />
      <h3>Compute them with the command line</h3>
      <p>
        Do this for any dataset that already exists in the workspace. Run from the repository root (that is where
        <code> image_sets/</code> lives, which is where datasets built from a server folder and tiled datasets keep their images).
      </p>
      <Steps>
        <li>Install the machine-learning extras once. The first run also downloads the model weights.<Cmd id="mlInstall" /></li>
        <li>Create the embeddings file <strong>next to the dataset&apos;s CSV</strong>, with the same name and a <code>.json</code> extension. Here for a dataset called <em>my-field-survey</em>
          (use <code>--device cuda</code> on a GPU; about 3 images per second on CPU):<Cmd id="embeddings" /></li>
        <li>Reload the portal. The dataset&apos;s panel now says it has embeddings and the Visual Explorer&apos;s <strong>Source → Embeddings</strong> button is enabled. No restart is needed.</li>
      </Steps>
      <Expect>
        <p>In the Visual Explorer choose <strong>Embeddings</strong> under Source, then a <strong>Projection</strong> (PCA is instant; t-SNE and LLE take longer on big sets) and
          <strong> Clusters</strong> (<em>Ground Truth</em> uses the folder-based clusters, <em>K-means</em> finds its own). The 3D map should show your images as points; drag to rotate and scroll to zoom.</p>
      </Expect>
      <Note title="Where the file must be">
        The dashboard finds embeddings by file name: the CSV <code>data_user/my-field-survey.csv</code> uses <code>data_user/my-field-survey.json</code>.
        Datasets in the Docker stack keep their CSVs in <code>precisionai/magnidata/data_user/</code> on your machine, so that is where the file goes.
        The tool resolves the image paths in the CSV from the folder you run it in: run it from the repository root for datasets whose images are
        under <code>image_sets/</code>, and from <code>precisionai/magnidata/</code> for datasets whose images are under <code>data_user/</code> (for example uploads).
      </Note>
      <h3>The file format</h3>
      <Output>{`{"embeddings": {"IMG_0001.jpg": [0.012, -0.4, ...], "IMG_0002.jpg": [0.31, 0.07, ...]}}`}</Output>
      <ul>
        <li>One entry per image, keyed by the image <strong>file name</strong> (no folders), so file names must be unique.</li>
        <li>Every vector has the same length. The tool&apos;s models give 384 numbers (<code>dinov2</code>, the default), 512 (<code>resnet18</code>) or 576 (<code>mobilenet</code>).</li>
        <li>Images without an entry simply do not appear in the 3D map.</li>
      </ul>
      <p>You can also bring vectors from your own model: write a file in this format and place it beside the CSV.</p>

      <h3>Other ways to attach them</h3>
      <ul>
        <li><strong>At creation, by upload.</strong> In <em>Create New Dataset</em> choose the embeddings file in the <strong>Embeddings</strong> field. Key each vector by the file
          name you are uploading; the portal renames uploaded images internally and re-keys your vectors to match. A file that matches none of the images is rejected with a clear message.</li>
        <li><strong>Permanent catalog entries.</strong> For a dataset registered in <code>confi.yaml</code> as <code>data/&lt;name&gt;.csv</code>, put <code>data/&lt;name&gt;.json</code> beside it.</li>
        <li><strong>Derived datasets inherit.</strong> A dataset saved from a selection, or from removals or a complement, uses its parent&apos;s embeddings. A merge keeps them only when every source uses the same file.
          Datasets made from <em>new</em> images, such as the tiles from the partition tool, need their own file.</li>
      </ul>
      <h3>Compare two models</h3>
      <p>Save a second file with a variant suffix, <code>&lt;name&gt;_&lt;variant&gt;.json</code>. The Visual Explorer then offers the variant for comparison against the default.</p>
      <Cmd id="embeddingsVariant" />
    </>
  )
}

function Reference() {
  return (
    <>
      <Goal>Look things up: data format, ports and files, and fixes for common problems.</Goal>
      <h3>Dataset format</h3>
      <p>A dataset is a CSV plus the files it points to:</p>
      <ul>
        <li><strong>Required</strong>: <code>image_path</code>, a path relative to the data root (for example <code>data_user/…/images/x.png</code>) or an allowed URL.</li>
        <li><strong>Recommended</strong>: <code>width</code>, <code>height</code>, and <code>cluster</code>.</li>
        <li><strong>Everything else</strong>: numeric columns become axes, filters and colours; text columns become categories.</li>
        <li><strong>Masks</strong>: <code>masks/&lt;name&gt;.png</code> next to <code>images/</code>, or a <code>mask_path</code> column. Black is background; each class has its own colour.</li>
        <li><strong>Embeddings (optional)</strong>: <code>&lt;name&gt;.json</code> beside the CSV, <code>{'{"embeddings": {"<image name>": [floats]}}'}</code>. It unlocks the 3D view; see chapter 10.</li>
      </ul>
      <p>To compute the feature columns for your own folder outside the portal:</p>
      <Cmd id="features" />
      <h3>Ports and places</h3>
      <table className="guide-table">
        <thead><tr><th>What</th><th>Where</th></tr></thead>
        <tbody>
          <tr><td>Portal</td><td><code>http://localhost:5175</code></td></tr>
          <tr><td>API</td><td><code>http://localhost:5051/api/health</code></td></tr>
          <tr><td>MLflow UI</td><td><code>http://localhost:5052</code></td></tr>
          <tr><td>Datasets and tiles</td><td><code>precisionai/magnidata/data_user/</code>, <code>image_sets/</code></td></tr>
          <tr><td>Runs, notes, partitions</td><td>Docker volumes (removed by <code>docker compose down -v</code>)</td></tr>
        </tbody>
      </table>
      <h3>Troubleshooting</h3>
      <table className="guide-table">
        <thead><tr><th>You see</th><th>Why, and what to do</th></tr></thead>
        <tbody>
          <tr><td>502 Bad Gateway, &quot;Dataset could not be restored&quot;</td><td>The API is still starting or crashed. Wait for it, then Retry; if it persists run <code>docker compose logs api</code>.</td></tr>
          <tr><td>Trainer card says Offline; Train is disabled</td><td>No worker has reported in. Check <code>docker compose ps</code> and the trainer log. After a fresh start it can take a few seconds.</td></tr>
          <tr><td>MLflow unavailable, Experiments empty</td><td>The <code>mlflow</code> service is not healthy. Restart the stack; runs are kept in its volume.</td></tr>
          <tr><td>A dataset says it is a Git LFS pointer</td><td>The CSV was cloned without its real content. Run <code>git lfs pull</code>, then reload.</td></tr>
          <tr><td>The graph is empty</td><td>No dataset is registered yet. Start with chapter 3 (Load Demo) or <strong>Add dataset</strong>.</td></tr>
          <tr><td>Merge fails with &quot;same CSV columns&quot;</td><td>The datasets were built differently. Merge datasets from the same pipeline.</td></tr>
          <tr><td><code>partition_dataset</code>: &quot;cannot reach the API&quot;</td><td>The stack is not running or listens elsewhere. Start it, or pass <code>--api</code>.</td></tr>
          <tr><td><code>partition_dataset</code>: &quot;source dataset not found&quot;</td><td>Run it from the repository root, or point <code>--data-root</code> at the folder that holds <code>data_user/</code>.</td></tr>
        </tbody>
      </table>
    </>
  )
}

export const CHAPTERS: readonly Chapter[] = [
  { id: 'overview', title: 'How the pieces fit', summary: 'Services, data flow and vocabulary', minutes: 4, Body: Overview },
  { id: 'install', title: 'Install and start', summary: 'Docker Compose, ports, GPU, reset', minutes: 5, Body: Install },
  { id: 'agristress', title: 'Get AgriStress-500', summary: 'Load the demo dataset', minutes: 3, Body: AgriStress },
  { id: 'partition', title: 'Partition it (command line)', summary: 'Tile and split into sub-datasets', minutes: 6, Body: Partition },
  { id: 'workspace', title: 'Tour the workspace', summary: 'Graph, panel, merge, badges', minutes: 5, Body: Workspace },
  { id: 'explorer', title: 'Explore a dataset', summary: 'Filters, previews, save a selection', minutes: 7, Body: Explorer },
  { id: 'train', title: 'Train and read results', summary: 'Runs, curves, overlays', minutes: 7, Body: Train },
  { id: 'loop', title: 'Close the loop', summary: 'Curate hard images and retrain', minutes: 6, Body: Loop },
  { id: 'own-data', title: 'Bring your own data', summary: 'Folder, upload or CSV', minutes: 6, Body: OwnData },
  { id: 'embeddings', title: 'Add embeddings', summary: 'Enable the 3D embeddings view', minutes: 5, Body: Embeddings },
  { id: 'reference', title: 'Reference and troubleshooting', summary: 'Formats, ports, fixes', minutes: 4, Body: Reference },
]
