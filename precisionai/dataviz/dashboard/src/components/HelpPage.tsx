import { useState } from 'react'
import './HelpPage.css'

interface Props {
  onOpenByod?: () => void
}

const SECTIONS = [
  { id: 'overview', label: 'What MagniData does' },
  { id: 'using-it', label: 'How to use it' },
  { id: 'data-format', label: 'Data format' },
  { id: 'custom-dataset', label: 'Creating a custom dataset' },
] as const

export function HelpPage({ onOpenByod }: Props) {
  const [active, setActive] = useState<string>(SECTIONS[0].id)

  const scrollTo = (id: string) => {
    setActive(id)
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div className="help-page">
      <aside className="help-toc">
        <div className="help-toc-label">On this page</div>
        <nav>
          {SECTIONS.map(s => (
            <button
              key={s.id}
              className={`help-toc-link${active === s.id ? ' help-toc-link--active' : ''}`}
              onClick={() => scrollTo(s.id)}
            >
              {s.label}
            </button>
          ))}
        </nav>
      </aside>

      <div className="help-content">
        <header className="help-page-header">
          <h1>Help</h1>
          <p>What MagniData does, how to use it, the data it expects, and how to build your own dataset for it.</p>
        </header>

        <section id="overview">
          <h2>What MagniData does</h2>
          <p>
            MagniData is a visual data-exploration and embedding-visualization workspace for
            image datasets. Instead of scrolling through thousands of thumbnails or eyeballing
            spreadsheet columns, it lets you look at an entire dataset at once — as points in a
            3D space, as filterable parallel-coordinate axes, or as a searchable table — and pivot
            between them without losing your place.
          </p>
          <p>Two complementary ways to look at the same images:</p>
          <ul>
            <li>
              <strong>Multiparametric features</strong> — deterministic, per-image measurements
              (exposure, blur, contrast, colorfulness, white balance, and — when the image has
              ground-truth annotations — object counts, class mix, and coverage). These are
              directly interpretable: you know exactly what each axis means.
            </li>
            <li>
              <strong>Embeddings</strong> — a dense vector per image, projected down to 3D (PCA,
              t-SNE, or LLE). Images that "read" similarly to whatever model produced the vectors
              end up near each other, surfacing groupings a human wouldn't think to filter for.
              MagniData doesn't compute embeddings itself — a dataset only has this view
              available when it (or you, via BYOD) brings pre-computed embeddings along.
            </li>
          </ul>
          <p>
            On top of either view: clustering (k-means, or a dataset's own ground-truth/committed
            clusters), brushing and filtering, per-image preview, and the ability to carve a
            selection out into a brand-new dataset.
          </p>
        </section>

        <section id="using-it">
          <h2>How to use it</h2>

          <h3>Getting to a dataset</h3>
          <p>
            From the landing screen, <strong>MagniData</strong> opens the dataset picker — a
            catalog of ready-made datasets plus any you've built yourself via BYOD (see below).
            Click a card to load it. You can also drop a raw CSV file onto the picker directly if
            you already have one in the right format.
          </p>

          <h3>The five views</h3>
          <p>Once a dataset is loaded, the tab bar switches between:</p>
          <ul>
            <li><strong>Visual Explorer</strong> — the 3D point cloud (Scatter or Clusters layout, Multiparametric or Embeddings source) alongside the parallel-coordinates panel.</li>
            <li><strong>Charts</strong> — distribution histograms for every numeric/categorical column.</li>
            <li><strong>Schema</strong> — a dictionary of every column: type, meaning, and summary stats. Available even before a dataset is loaded.</li>
            <li><strong>Table</strong> — a sortable, filterable row browser with an inline preview button per row.</li>
            <li><strong>Overview</strong> — dataset-level summary statistics.</li>
          </ul>

          <h3>Key interactions</h3>
          <ul>
            <li><strong>Click a point</strong> (3D view) or the play button (Table) to open the image preview, with quality/feature stats alongside it.</li>
            <li><strong>Ctrl/⌘+click a point</strong> to select its whole cluster; pick a cluster from the dropdown to do the same without clicking.</li>
            <li><strong>Brush an axis</strong> in parallel coordinates to filter rows by a range or category — brushes combine (AND) across axes and stay in sync with the 3D view and table.</li>
            <li><strong>Create partitions</strong> writes the current k-means clustering back into the dataset's cluster column for the selected rows, so it persists across sessions.</li>
            <li><strong>Save a curated subset</strong> — remove or hand-pick images in the table/preview, then save the result as a new, independent dataset (it inherits the parent's embeddings automatically).</li>
          </ul>
        </section>

        <section id="data-format">
          <h2>Data format</h2>
          <p>
            Every dataset is a CSV: one row per image. Two columns are load-bearing, everything
            else is optional and additive.
          </p>

          <h3>Required</h3>
          <ul>
            <li>
              <code>image_path</code> — a path to the image, relative to the data root
              (e.g. <code>image_sets/my_set/images/img001.jpg</code>). MagniData auto-detects
              this column by name, so it must contain <code>image</code> and <code>path</code>.
            </li>
          </ul>

          <h3>Recommended</h3>
          <ul>
            <li><code>cluster</code> / <code>cluster_l2</code> — a ground-truth grouping (e.g. by site, batch, or condition), shown as "Ground Truth" clusters wherever the dataset has one.</li>
            <li><code>width</code>, <code>height</code> — image dimensions, used by the preview and a few derived features.</li>
          </ul>

          <h3>Everything else — feature columns</h3>
          <p>
            Any additional numeric or categorical column becomes a first-class axis: a parallel-
            coordinates dimension, a histogram, a color-by option, a 3D multiparametric axis.
            Datasets built through BYOD (below) come with a standard set covering exposure, focus,
            contrast, colorfulness, white balance, and — when annotations were provided — object
            coverage, instance counts, and class mix. Open the <strong>Schema</strong> tab on any
            loaded dataset to see the exact columns it has and what each one means.
          </p>

          <h3>Embeddings (optional, unlocks the Embeddings source)</h3>
          <p>
            MagniData never computes embeddings itself — bring your own. A sibling JSON file
            named after the CSV's stem (<code>my_set.csv</code> → <code>my_set.json</code>) with
            the shape:
          </p>
          <pre><code>{'{ "embeddings": { "<image-basename>": [0.12, -0.04, …] } }'}</code></pre>
          <p>
            Keys are image <em>basenames</em> (not full paths), matched to rows by filename. A
            second model's vectors against the same images can live alongside as{' '}
            <code>my_set_&lt;model&gt;.json</code> and shows up as a "Compare" overlay.
          </p>
        </section>

        <section id="custom-dataset">
          <h2>Creating a custom dataset</h2>
          <p>
            You don't need to hand-produce any of the above — <strong>BYOD → Create New
            Dataset</strong> runs the whole pipeline for you, with progress shown in the header
            until it's done.
          </p>

          <ol>
            <li>
              <strong>Name it</strong>, and optionally add a description.
            </li>
            <li>
              <strong>Add your images</strong> — pick a folder or a set of files (JPG, PNG, BMP,
              WEBP, or TIFF). Selecting a folder preserves its subfolder structure as the{' '}
              <code>cluster</code>/<code>cluster_l2</code> grouping (e.g.{' '}
              <code>siteA/batch1/img.jpg</code> → cluster <code>siteA/batch1</code>); a flat
              selection is grouped as a single "uncategorized" cluster. Up to 500 images / 200MB
              per build.
            </li>
            <li>
              <strong>Add annotations (optional)</strong> — one COCO-format JSON file per image,
              matched by filename (<code>img001.jpg</code> ↔ <code>img001.json</code>): the
              standard <code>images</code>/<code>categories</code>/<code>annotations</code> shape,
              with a polygon <code>segmentation</code> and/or <code>bbox</code> per instance.
              Images without a matching file still build fine — they simply get quality/exposure
              columns instead of instance/coverage/class columns.
            </li>
            <li>
              <strong>Add embeddings (optional)</strong> — MagniData doesn't compute embeddings, so
              if you want the Embeddings view for this dataset, bring your own pre-computed vectors
              as a single JSON file in the shape described under Data Format above. Without one,
              the dataset builds fine and works normally — it just won't have an Embeddings source.
            </li>
            <li>
              <strong>Submit.</strong> MagniData stages the images, computes the quality/exposure
              (and, where annotated, instance/coverage/class) feature columns for each one, stages
              your embeddings file if you provided one, and registers the result as a new dataset —
              it appears in the MagniData picker as soon as it's done, ready to explore like any
              other.
            </li>
          </ol>

          <p>
            Prefer to see it work first? <strong>BYOD → Load Demo</strong> builds the COCO128
            sample dataset the same way, with no upload required (demo builds don't include
            embeddings either — the pre-registered "COCO128" dataset already in the picker has
            them, if you want to see that view).
          </p>

          {onOpenByod && (
            <button className="help-cta" onClick={onOpenByod}>Open BYOD</button>
          )}
        </section>
      </div>
    </div>
  )
}
