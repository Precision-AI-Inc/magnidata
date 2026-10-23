# task-27 — Interactive dataset lineage viewport

**Status:** completed

**Depends on:** task-26
**Commit message:** `Make dataset lineage graph interactive`

## Description

Replace the clipped, fixed-position dataset lineage canvas with an interactive viewport so
large and deep dataset graphs remain explorable on desktop and touch devices.

## Proposed solution

- Render the lineage with React Flow while retaining the existing dataset cards and actions.
- Support drag-to-pan, wheel and pinch zoom, fit-to-view controls, and a pannable minimap.
- Keep the graph read-only and use a deterministic left-to-right lineage layout.
- Theme nodes, edges, controls, and the minimap consistently in light and dark modes.
- Preserve dataset loading, loading state, parent summaries, and deletion confirmation.

## Testing

- Test deterministic lineage coordinates and spacing for wide and deep graphs.
- Run the dashboard production build and test suite plus repository gates.
