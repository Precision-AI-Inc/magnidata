# task-26 — Cloud dataset graph and complement curation

**Status:** completed (`f9bde14`)

**Depends on:** task-25
**Commit message:** `Add cloud complement dataset curation`

## Description

Treat the two AgriStress mini datasets as children of a combined, trainable Cloud Dataset.
Show dataset lineage as a graph and let filtered hard samples drive a complement search over
the cloud pool. Image removal from preview remains non-destructive and is materialized only
when the user creates a derived dataset.

## Proposed solution

- Support multiple dataset parents while preserving existing single-parent records.
- Create and register Cloud Dataset as the union of AgriStress500-mini1 and mini2, with both
  mini datasets linked beneath it.
- Render registered datasets as a selectable lineage graph.
- Add a right-panel Complement action that selects the top-K nearest cloud-complement rows
  in standardized shared numeric feature space.
- Combine staged removals and complement additions in one derived dataset. Complement results
  have both the current dataset and Cloud Dataset as parents; removal-only results have only
  the current dataset as parent.
- Add a Remove image action to image preview that stages a non-destructive deletion.

## Testing

- Test multi-parent lineage migration and cleanup.
- Test Cloud Dataset union creation and mini-dataset edges.
- Test complement selection, exclusion of current rows, deterministic ranking, and derived
  dataset parent edges.
- Test dataset graph layout and complement UI helpers.
- Run the frontend production build and test suite plus repository gates.
