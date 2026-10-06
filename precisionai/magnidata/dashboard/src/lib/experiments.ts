import type { RunStatus, RunSummary } from '../api'

export interface MetricColumns {
  metrics: string[]
  params: string[]
}

/** Union of the metric and param keys seen across runs, each sorted. */
export function metricColumns(runs: RunSummary[]): MetricColumns {
  const metrics = new Set<string>()
  const params = new Set<string>()
  for (const run of runs) {
    Object.keys(run.metrics).forEach(k => metrics.add(k))
    Object.keys(run.params).forEach(k => params.add(k))
  }
  return { metrics: [...metrics].sort(), params: [...params].sort() }
}

/** "42 s", "3 min 05 s", "2 h 07 min"; runs without an end time count up to `now`. */
export function formatDuration(run: Pick<RunSummary, 'start_time' | 'end_time' | 'duration_ms'>, now = Date.now()): string {
  const ms = run.duration_ms ?? (run.start_time && !run.end_time ? now - run.start_time : null)
  if (ms == null || ms < 0) return '—'
  const total = Math.round(ms / 1000)
  if (total < 60) return `${total} s`
  const minutes = Math.floor(total / 60)
  if (minutes < 60) return `${minutes} min ${String(total % 60).padStart(2, '0')} s`
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, '0')} min`
}

/** "GPU" for a cuda device tag, "CPU" for cpu, null when the run never reported a device. */
export function deviceBadge(device: string | undefined): 'GPU' | 'CPU' | null {
  if (!device) return null
  return device.startsWith('cuda') ? 'GPU' : 'CPU'
}

/** "ETA 11 min" from the `magnidata.eta_seconds` tag; only meaningful while the run is RUNNING. */
export function formatEta(status: string, etaSeconds: string | undefined): string | null {
  if (status !== 'RUNNING' || etaSeconds == null || etaSeconds === '') return null
  const total = Number(etaSeconds)
  if (!Number.isFinite(total) || total < 0) return null
  if (total < 60) return `ETA ${Math.round(total)} s`
  const minutes = Math.round(total / 60)
  if (minutes < 60) return `ETA ${minutes} min`
  return `ETA ${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, '0')} min`
}

/** Device badge and ETA of a run as one short label, e.g. "GPU · ETA 11 min". */
export function deviceLabel(run: Pick<RunSummary, 'status' | 'magnidata'>): string {
  const badge = deviceBadge(run.magnidata['magnidata.device'])
  const eta = formatEta(run.status, run.magnidata['magnidata.eta_seconds'])
  return [badge, eta].filter(Boolean).join(' · ') || '—'
}

export type StatusTone = 'neutral' | 'info' | 'success' | 'danger' | 'warning'

export function statusTone(status: RunStatus | string): StatusTone {
  switch (status) {
    case 'FINISHED': return 'success'
    case 'RUNNING': return 'info'
    case 'SCHEDULED': return 'warning'
    case 'FAILED': return 'danger'
    default: return 'neutral'
  }
}

export interface FilterState {
  status?: RunStatus | null
  dataset?: string | null
  text?: string
}

const CLAUSE_HINT = /[=<>]|\bLIKE\b/i

/** Free text that already looks like an MLflow clause is passed through; plain words search run names. */
function textClause(text: string): string | null {
  const t = text.trim()
  if (!t) return null
  if (CLAUSE_HINT.test(t)) return t
  return `attributes.run_name LIKE '%${t.replace(/['"`%]/g, '')}%'`
}

/** Combine the status chip, dataset toggle and free text into one MLflow filter string. */
export function buildFilter(state: FilterState): string {
  const clauses: string[] = []
  if (state.status) clauses.push(`attributes.status = '${state.status}'`)
  if (state.dataset) clauses.push(`tags.\`magnidata.dataset_source\` = '${state.dataset.replace(/'/g, '')}'`)
  const text = textClause(state.text ?? '')
  if (text) clauses.push(text)
  return clauses.join(' AND ')
}

export type ColumnKind = 'attribute' | 'metric' | 'param'

/** MLflow `order_by` entry for a sortable column, or null when the column cannot be sorted server-side. */
export function buildOrderBy(kind: ColumnKind, key: string, dir: 'asc' | 'desc'): string | null {
  const suffix = dir === 'asc' ? 'ASC' : 'DESC'
  if (kind === 'attribute') {
    const attr: Record<string, string> = { name: 'run_name', start: 'start_time', status: 'status' }
    return attr[key] ? `attributes.${attr[key]} ${suffix}` : null
  }
  return `${kind === 'metric' ? 'metrics' : 'params'}.\`${key}\` ${suffix}`
}

export const DEFAULT_METRICS = ['val_loss/loss', 'val_metric/miou', 'train_loss/loss']
export const DEFAULT_PARAMS = ['epochs', 'lr']

/** Experiments to select initially: every `magnidata/` one plus the most recently updated few. */
export function defaultSelection<T extends { experiment_id: string; name: string; last_update_time: number }>(
  experiments: T[], recent = 3,
): string[] {
  const picked = new Set(experiments.filter(e => e.name.startsWith('magnidata/')).map(e => e.experiment_id))
  ;[...experiments].sort((a, b) => b.last_update_time - a.last_update_time).slice(0, recent).forEach(e => picked.add(e.experiment_id))
  return [...picked]
}

export function formatMetric(value: number | undefined): string {
  if (value == null || Number.isNaN(value)) return '—'
  if (Number.isInteger(value)) return value.toLocaleString()
  return Math.abs(value) < 1e-3 ? value.toExponential(2) : value.toPrecision(4).replace(/\.?0+$/, '')
}
