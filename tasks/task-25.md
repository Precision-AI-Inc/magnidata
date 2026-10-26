# task-25 — Restore the last dataset and keep large visualizations responsive

**Status:** completed (`a94a238`)

**Depends on:** task-24
**Commit message:** `Restore datasets and sample large parallel plots`

## Description

Refreshing the dashboard forgets the active server dataset, and drawing every row of a
large dataset in Plotly parallel coordinates can freeze the explorer. Preserve the user's
last dataset context and bound visualization work without changing filtering semantics.

## Proposed solution

- Remember the most recently loaded server dataset in local storage and restore it after
  the dataset catalogue becomes available following a browser refresh.
- Retry catalogue and CSV requests briefly so an API that is busy or restarting does not
  immediately discard the remembered dataset.
- Surface restoration failures while keeping the dataset picker usable.
- Publish coarse preparation and batch progress through MLflow so the training badge updates
  smoothly and distinguishes caching/indexing from GPU epochs.
- Draw a deterministic, representative sample of large parallel-coordinate datasets while
  applying brushes, counts, tables, exports, and downstream selections to all rows.
- Throttle plot-height updates during splitter dragging.

## Testing

- Unit-test dataset-memory/retry helpers and deterministic plot sampling.
- Verify refresh restores the latest dataset and explicit navigation still works.
- Verify training-stage progress is formatted for queued, preparing, indexing, and epoch states.
- Verify a 50k+ row dataset renders a bounded number of parallel-coordinate lines while
  filters continue to operate over the complete dataset.
- Run the frontend production build and test suite.
