# task-30 — Experiment badge on lineage dataset cards

**Status:** completed

**Depends on:** task-29
**Commit message:** `Show experiment status on dataset cards`

## Description

Mark each dataset in the lineage graph with an experiment icon beside the dataset icon so it is clear
at a glance which datasets have training activity.

## Proposed solution

- **Yellow** experiment icon when the dataset has live experiments (runs that are `RUNNING` or
  `SCHEDULED`); **gray** when it has none. Datasets that only have finished, failed or killed runs show
  gray with the run count in the tooltip (e.g. "3 experiments, none running") — assumed, please confirm.
- A pulsing ring on the yellow badge, disabled under `prefers-reduced-motion`; badge has an accessible
  label and a tooltip ("2 running · 5 total" / "No experiments").
- Per-dataset counts come from one request, not one per card: add `GET /api/datasets/experiment-counts`
  returning `{source: {total, active}}` aggregated from MLflow runs by the
  `magnidata.dataset_source` tag. Return an empty map (not an error) when MLflow is unreachable.
- The workspace refreshes the counts when it mounts and while any dataset is active (every ~10 s),
  matching the Experiments view's live refresh.
- Merged and selection-derived datasets (task-29) naturally show the gray badge.

## Testing

- API tests for the aggregation (active vs finished, unlinked runs ignored, MLflow down returns `{}`).
- Unit tests for the badge state helper (yellow/gray/tooltip text).
- Browser check: start a run and watch the badge turn yellow, then gray when it finishes.
- Run the dashboard build and test suite plus repository gates.

## Implementation notes

- `count_runs_by_dataset` / `GET /api/datasets/experiment-counts` (`api/experiments.py`) page through MLflow runs
  (capped at 2,000) and group by the `magnidata.dataset_source` tag.
- `useExperimentCounts` polls every 10 s only while a run is active; `experimentBadge` (`lib/experimentBadge.ts`)
  chooses yellow/gray and the tooltip. Verified in a browser against a scratch MLflow store with a RUNNING and a
  FINISHED run.
