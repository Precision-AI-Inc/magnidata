# task-24 — Calm Material-style dashboard shell

**Depends on:** task-23
**Commit message:** `Simplify the dashboard with a Material-style workspace`

## Description

The dashboard exposes every workflow, dataset statistic, service state and action at the
same visual weight. This makes the workspace feel dense before a user has chosen a task.
Use the information hierarchy from Material Dashboard React as a reference while keeping
MagniData's existing design tokens and all current workflows.

## Proposed solution

- Turn the primary navigation into a floating, rounded sidebar with clearer section labels,
  stronger active state, and a compact workspace status footer.
- Simplify the context header into breadcrumbs, page title, small dataset context chips and
  focused global actions.
- Rework the workspace around four summary cards, a prominent dataset panel, a compact
  recent-runs panel and three task-oriented shortcuts.
- Put detailed service readiness behind progressive disclosure so it remains available
  without competing with primary content.
- Keep the connected three-pane Visual Explorer canvas, but move its dataset views into the
  left navigation as children of Visual Explorer.
- Move training, run overlay, colour, sorting, field and selection actions into a collapsible
  right configuration panel so the canvas gets more vertical room.
- Keep dataset identity and sample counts in the page header while reserving the far-right
  header action for the light/dark theme toggle.
- Add responsive layouts for narrow desktops and mobile-width screens without removing
  navigation or actions.
- Let the image preview queue inference when a run has no saved prediction for the current
  sample. Run inference in the trainer worker, persist the mask as an MLflow artifact, poll
  its status in the modal, and reuse the saved preview on later loads.

## Testing

- `cd precisionai/magnidata/dashboard && npm run build && npm test`.
- Verify the workspace at desktop and narrow widths, in light and dark themes.
- Verify navigation, dataset opening, upload, experiment and help actions remain reachable.
- Verify nested dataset navigation and the collapsed/expanded configuration panel.
- Verify missing predictions can be queued, generated, displayed, and loaded from the saved
  artifact without rerunning inference.
