# task-28 — Unified workspace: lineage graph with dataset detail panel

**Status:** completed

**Depends on:** task-27
**Commit message:** `Unify the dashboard and datasets into a lineage workspace`

## Description

Task-27 was marked completed but the lineage graph was not usable: React Flow sets
`pointer-events: none` on nodes that are neither selectable nor draggable, so the card's open
and delete buttons never received clicks. Selecting a dataset therefore could not open the
Visual Explorer, and datasets could not be deleted. Fix that, then merge the landing dashboard
and the datasets view into one workspace built around the lineage graph.

## Proposed solution

### Bug fix (carried over from task-27)

- Keep `pointer-events: all` on `.dataset-graph__node` (already in the working tree) so the card
  buttons receive clicks.
- Make nodes selectable so a click selects a dataset instead of immediately loading it; loading
  moves to an explicit action in the detail panel. Delete stays on the card, behind confirmation.
- Verify in a real browser (docker compose stack or `npm run dev` plus API): select, delete,
  and open in Visual Explorer all work. Build and unit tests alone missed this bug.

### One workspace view

Merge `WorkspaceDashboard` (landing) and `CSVLoader` (datasets) into a single view, top to bottom:

1. **Global summary strip** — storage-wide and dataset-wide totals: dataset count, total
   samples, images and storage footprint, annotation counts, runs and experiments, trainer/GPU
   status. Reuse what `WorkspaceDashboard` already shows.
2. **Lineage graph** — the interactive React Flow graph, filling the main area. The selected
   node is highlighted and its ancestors and descendants stay emphasized while the rest dim.
3. **Dataset detail panel** (right, docked beside the graph; becomes a bottom sheet on narrow
   screens) — shown when a dataset is selected, with tabs or sections for:
   - Statistics: row count, columns, split sizes, key numeric distributions as small sparklines.
   - Images: thumbnail strip or grid sampled from the dataset, opening the image preview.
   - Annotations: counts and class breakdown.
   - Runs and experiments: recent runs trained on this dataset with status and headline metric.
   - Lineage: parents, children, and how the dataset was derived (removals, complement).
4. **Primary actions in the panel header**
   - **Open in Visual Explorer** — loads the dataset and switches to the explorer.
   - **View experiments** — opens the Experiments view pre-filtered to this dataset.
   - Delete (user-created datasets only), with confirmation.

Empty state (nothing selected) shows a short hint and the global summary only. Retain the upload-CSV
and BYOD entry points in a compact toolbar above the graph.

### Polish ("make it cool")

- Smooth panel slide-in, subtle node selection ring, animated edges along the highlighted lineage.
- Minimap and controls stay; panel and graph resize together without layout shift.
- Keyboard: arrow keys move selection between connected nodes, Enter opens the explorer, Esc
  clears selection. Respect light and dark themes and `prefers-reduced-motion`.
- Persist the selected dataset in the URL or local state so reload restores it.

## Backend

- Add a per-dataset summary endpoint (stats, annotation counts, recent runs) if the existing
  endpoints cannot supply the panel without many round trips. Keep responses cheap; sample
  images rather than listing all.
- Add a storage-wide summary endpoint if global totals are not already available.
- Accept a dataset filter on the experiments listing used by "View experiments".

## Testing

- Unit tests for selection state, lineage highlighting (ancestors/descendants), and keyboard
  navigation helpers.
- Component test that the detail panel actions call open, view-experiments, and delete handlers.
- Python tests for any new summary endpoints and the experiments dataset filter.
- Manual browser check (or a Playwright/devtools run) of select → open in Visual Explorer,
  select → view experiments, and delete; record the result in the task before marking completed.
- Run the dashboard production build and test suite plus repository gates.

## Implementation notes

- `WorkspaceView` replaces `WorkspaceDashboard` and `CSVLoader`; the separate Datasets navigation entry is gone.
- `DatasetGraph` selection, lineage emphasis and keyboard navigation use `lineageOf` and `adjacentDataset`
  (`lib/datasetGraph.ts`). The selection is stored in `localStorage` (`magnidata.workspace.selected`).
- `GET /api/datasets/summary` (`api/dataset_summary.py`) feeds the panel; configured datasets now report `row_count`.
  Runs for the panel come from the existing MLflow endpoints filtered by dataset, so no experiments filter was added to
  the API; `ExperimentsTab` takes a `datasetFilter` instead.
- Image thumbnails open the full-size image in a new tab rather than the Visual Explorer's image preview, which is
  coupled to loaded rows.
- The dashboard has no DOM test environment, so panel action wiring is covered by the browser check below, not a
  component test.

## Verification (browser, synthetic data root)

- Click a card: it is selected, the panel opens and the node is centred; reload restores the selection.
- Open in Visual Explorer loads the dataset and switches to the explorer.
- View experiments opens Experiments with "Only runs for <dataset>" checked.
- Delete on a derived dataset's card removed it (`DELETE /api/datasets/<id>`, graph and totals refreshed).
- Arrow keys moved selection, Escape cleared it.
