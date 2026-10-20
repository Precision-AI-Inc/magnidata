# task-29 — Create datasets from explorer selections and by merging datasets

**Status:** completed

**Depends on:** task-28
**Commit message:** `Create datasets from selections and merges`

## Description

The Visual Explorer can only export the current selection as a CSV download, so a selection never
appears in the lineage graph. Let a selection become a registered dataset, and let the workspace
merge several existing datasets into a new one. Neither new dataset has any experiments.

## Proposed solution

### Dataset from a selection (Visual Explorer)

- Next to **Export selection**, add **Save as dataset**: name (and optional description) inline, same
  pattern as the existing curated-removals form. It saves the rows currently shown (filters, brushes
  and staged removals applied) as a child of the loaded dataset.
- Reuse `POST /api/datasets` (`parent_source`, `image_names`); no new endpoint is needed. After saving,
  reload the catalogue and show a confirmation with a link that opens the workspace with the new dataset
  selected in the lineage graph.
- Keep CSV export as is. Disable the action when the selection is empty or the dataset has no image
  column.

### Merge datasets (Workspace)

- Multi-select in the lineage graph (Shift/Ctrl-click) and a **Merge datasets** toolbar action that is
  enabled for two or more selected datasets; the detail panel shows the selection count while several
  are selected.
- New `POST /api/datasets/merge` `{name, description?, sources:[...]}`: writes the de-duplicated union with
  `dataset_curation.combine_sources` (rows keyed by image name), registers every source as a parent, and
  returns the new dataset record (201). Reject fewer than two sources, unknown sources, a name that is
  empty, and sources whose columns differ (400 with the existing "same CSV columns" message).
- The merged dataset is created with no runs, so it shows no experiments until one is trained on it.
- Embeddings: inherit the first parent's embeddings source only when every parent resolves to the same
  one; otherwise the merged dataset has none.

## Testing

- API tests: merge happy path (union, de-duplication, all parent edges, `deletable`), validation errors,
  column mismatch, and a merged dataset reporting zero runs.
- Frontend tests for the multi-selection helpers; a browser check of save-selection (appears in the
  graph) and merge (new node with edges from every source).
- Run the dashboard build and test suite plus repository gates.

## Implementation notes

- Save as dataset reuses `POST /api/datasets`; merge is `POST /api/datasets/merge` (`api/app.py`,
  `dataset_curation.parse_merge_request`), sharing a `_write_child_csv` helper with dataset creation.
- Multi-selection lives in `WorkspaceView` (`lib/workspaceSelection.ts`); the merge form is the side panel
  (`MergePanel`) that opens when two or more datasets are selected, so there is no separate toolbar button.
- After saving a selection, "View in workspace" stores the new dataset as the workspace selection and navigates home.
- Verified in a browser: Shift-click Root Set + Sub Set, merge -> node with two parent edges and a gray badge;
  Save as dataset from the explorer -> new child node selected in the workspace.
