import type { RunSummary } from '../api'
import { deviceBadge } from './experiments'

/** Training columns worth showing as axes as soon as a run is overlaid, in display order. */
export const TRAINING_HEADLINE = ['loss_total', 'loss_total_mean', 'hardness']
export const COMPARISON_HEADLINE = ['cmp_hardness_change', 'cmp_loss_total_change']
/** A sample counts as persistently hard from this hardness up. */
export const HIGH_HARDNESS = 0.6
const TOP_FRACTION = 0.2

/**
 * The axes to show after a run overlay loads: the current ones plus the headline training columns
 * that exist (`available`) and came from the run (`runColumns`), each added once, at the end.
 */
export function defaultTrainingAxes(
  available: Iterable<string>, current: Iterable<string>, runColumns: readonly string[], headline: readonly string[] = [...TRAINING_HEADLINE, ...COMPARISON_HEADLINE],
): string[] {
  const have = new Set(available)
  const out = [...current]
  const seen = new Set(out)
  for (const col of headline) {
    if (have.has(col) && runColumns.includes(col) && !seen.has(col)) { out.push(col); seen.add(col) }
  }
  return out
}

export interface Brush { col: string; range: [number, number] }

/**
 * The brush behind "High-loss samples": hardness at or above `HIGH_HARDNESS`, or, when the run has no
 * hardness, the top fifth of `loss_total_mean`. Null when neither column holds numbers.
 */
export function highLossBrush(columns: readonly string[], rows: readonly Record<string, unknown>[]): Brush | null {
  const values = (col: string) => rows.map(r => r[col]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  if (columns.includes('hardness')) {
    const v = values('hardness')
    if (v.length) return { col: 'hardness', range: [HIGH_HARDNESS, Math.max(HIGH_HARDNESS, ...v)] }
  }
  if (columns.includes('loss_total_mean')) {
    const v = values('loss_total_mean').sort((a, b) => a - b)
    if (v.length) return { col: 'loss_total_mean', range: [v[Math.min(v.length - 1, Math.floor(v.length * (1 - TOP_FRACTION)))], v[v.length - 1]] }
  }
  return null
}

export interface PickerEntry {
  run: RunSummary
  selectable: boolean
  /** Why the run cannot be picked (disabled), or `split only` for a pickable run that has no losses yet. */
  reason: string | null
  scored: number
  totalEpochs: number
  device: 'GPU' | 'CPU' | null
}

function scoredCount(run: RunSummary): number {
  return run.magnidata['magnidata.scored_epochs']?.split(',').filter(Boolean).length ?? 0
}

function entry(run: RunSummary): PickerEntry {
  const scored = scoredCount(run)
  const base = {
    run, scored, totalEpochs: Number(run.params.epochs) || 0, device: deviceBadge(run.magnidata['magnidata.device']),
  }
  if (run.has_per_sample) return { ...base, selectable: true, reason: null }
  if (run.has_split) return { ...base, selectable: true, reason: 'split only' }
  switch (run.status) {
    case 'SCHEDULED': return { ...base, selectable: false, reason: 'queued' }
    case 'RUNNING': return { ...base, selectable: false, reason: 'training · first losses after epoch 1 scoring' }
    case 'FAILED': return { ...base, selectable: false, reason: 'failed — no per-sample data' }
    case 'KILLED': return { ...base, selectable: false, reason: 'cancelled — no per-sample data' }
    default: return { ...base, selectable: false, reason: 'no per-sample data' }
  }
}

/** Every run linked to the dataset, newest first, each marked pickable or not (with the reason). */
export function pickerEntries(runs: readonly RunSummary[]): PickerEntry[] {
  const started = (r: RunSummary) => r.start_time ?? Number.MAX_SAFE_INTEGER
  return runs.map(entry).sort((a, b) => started(b.run) - started(a.run))
}

/** One-line text for a picker option, e.g. `my run · running · GPU · scored 2/10 · split only`. */
export function pickerLabel(e: PickerEntry): string {
  const parts = [e.run.name || e.run.run_id.slice(0, 8), e.run.status.toLowerCase()]
  if (e.device) parts.push(e.device)
  if (e.totalEpochs && e.run.status === 'RUNNING') parts.push(`scored ${e.scored}/${e.totalEpochs}`)
  else if (e.scored) parts.push(`${e.scored} epochs`)
  if (e.reason) parts.push(e.reason)
  return parts.join(' · ')
}

/** The run to overlay automatically: the launched one, else the newest running one, once it can be picked. */
export function autoOverlayTarget(entries: readonly PickerEntry[], launchedRunId: string | null, already: ReadonlySet<string>): PickerEntry | null {
  const fresh = (e: PickerEntry) => e.selectable && !already.has(e.run.run_id)
  const launched = launchedRunId ? entries.find(e => e.run.run_id === launchedRunId) : undefined
  if (launched) return fresh(launched) ? launched : null
  return entries.find(e => e.run.status === 'RUNNING' && fresh(e)) ?? null
}
