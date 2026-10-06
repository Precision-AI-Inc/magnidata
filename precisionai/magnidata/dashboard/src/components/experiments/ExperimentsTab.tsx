import { useMemo, useState } from 'react'
import { AlertTriangle, ChevronDown, ChevronUp, ChevronsUpDown, ExternalLink, Loader2, RefreshCw, Search } from 'lucide-react'
import type { RunStatus, RunSummary } from '../../api'
import { useExperiments } from '../../hooks/useExperiments'
import type { SortState } from '../../hooks/useExperiments'
import { formatDuration, formatMetric, metricColumns, statusTone } from '../../lib/experiments'
import type { ColumnKind } from '../../lib/experiments'
import { TABLE_HEADER, TABLE_HEADER_ACTIVE } from '../../data/vizColors'
import { RunDetail } from './RunDetail'

interface Props {
  /** Dataset CSV path relative to the data root, when one is loaded. */
  datasetSource: string | null
  selectedRunId: string | null
  onSelectRun: (run: RunSummary) => void
  onCloseRun: () => void
  knownDatasets: string[]
  onLoadDataset: (source: string) => void
}

const STATUSES: RunStatus[] = ['SCHEDULED', 'RUNNING', 'FINISHED', 'FAILED', 'KILLED']

const TONE_STYLE: Record<string, { bg: string; fg: string }> = {
  success: { bg: 'var(--color-success-bg)', fg: 'var(--color-success-text)' },
  danger: { bg: 'var(--color-danger-bg)', fg: 'var(--color-danger)' },
  info: { bg: 'var(--c-accent-light)', fg: 'var(--c-accent)' },
  warning: { bg: 'var(--c-raised)', fg: 'var(--c-t2)' },
  neutral: { bg: 'var(--c-raised)', fg: 'var(--c-t3)' },
}

export function StatusBadge({ status }: { status: string }) {
  const tone = TONE_STYLE[statusTone(status)]
  return (
    <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold"
          style={{ background: tone.bg, color: tone.fg, border: '1px solid var(--c-border)' }}>
      {status}
    </span>
  )
}

export function ExperimentsTab({ datasetSource, selectedRunId, onSelectRun, onCloseRun, knownDatasets, onLoadDataset }: Props) {
  const x = useExperiments(datasetSource)
  const [pickerOpen, setPickerOpen] = useState(false)
  const seen = useMemo(() => metricColumns(x.runs), [x.runs])
  const experimentNames = useMemo(() => new Map(x.experiments.map(e => [e.experiment_id, e.name])), [x.experiments])
  const down = x.mlflow != null && !x.mlflow.reachable

  const sortBy = (kind: ColumnKind, key: string) => {
    const next: SortState = x.sort.kind === kind && x.sort.key === key
      ? { kind, key, dir: x.sort.dir === 'desc' ? 'asc' : 'desc' }
      : { kind, key, dir: 'desc' }
    x.setSort(next)
  }

  const toggleColumn = (kind: 'metrics' | 'params', key: string) => {
    const has = x.columns[kind].includes(key)
    x.setColumns({ ...x.columns, [kind]: has ? x.columns[kind].filter(k => k !== key) : [...x.columns[kind], key] })
  }

  return (
    <div className="flex h-full" style={{ minHeight: 0 }}>
      {/* Experiment list */}
      <aside className="flex flex-col shrink-0" style={{ width: 240, borderRight: '1px solid var(--c-border)' }}>
        <div className="px-3 py-2.5" style={{ borderBottom: '1px solid var(--c-border)' }}>
          <div className="text-xs font-semibold mb-2" style={{ color: 'var(--c-t2)' }}>Experiments</div>
          <label className="flex items-center gap-1.5 px-2 py-1 rounded-lg"
                 style={{ background: 'var(--c-raised)', border: '1px solid var(--c-border)' }}>
            <Search size={12} style={{ color: 'var(--c-t3)' }} />
            <input value={x.experimentQuery} onChange={e => x.setExperimentQuery(e.target.value)}
                   placeholder="Search experiments" aria-label="Search experiments"
                   className="text-xs outline-none flex-1 bg-transparent" style={{ color: 'var(--c-t1)', minWidth: 0 }} />
          </label>
        </div>
        <div className="flex-1 overflow-auto py-1">
          {x.experiments.length === 0 && (
            <p className="text-xs px-3 py-3" style={{ color: 'var(--c-t3)' }}>No experiments yet.</p>
          )}
          {x.experiments.map(e => (
            <label key={e.experiment_id} className="flex items-center gap-2 px-3 py-1.5 cursor-pointer text-xs"
                   style={{ color: 'var(--c-t1)' }}>
              <input type="checkbox" checked={x.selected.includes(e.experiment_id)}
                     onChange={() => x.toggleExperiment(e.experiment_id)} />
              <span className="truncate" title={e.name}>{e.name}</span>
            </label>
          ))}
        </div>
      </aside>

      <section className="flex flex-col flex-1" style={{ minWidth: 0 }}>
        {down && (
          <div role="alert" className="flex items-center gap-2 px-4 py-2 text-xs font-semibold"
               style={{ background: 'var(--color-danger-bg)', color: 'var(--color-danger)', borderBottom: '1px solid var(--color-danger)' }}>
            <AlertTriangle size={14} />
            MLflow is not reachable{x.mlflow?.tracking_uri ? ` at ${x.mlflow.tracking_uri}` : ''}. Check that the mlflow service is running.
          </div>
        )}

        {/* Toolbar */}
        <div className="flex items-center gap-2 px-4 py-2.5 shrink-0 flex-wrap" style={{ borderBottom: '1px solid var(--c-border)' }}>
          <div className="flex items-center gap-1" role="group" aria-label="Status filter">
            {([null, ...STATUSES] as (RunStatus | null)[]).map(s => (
              <button key={s ?? 'all'} onClick={() => x.setStatus(s)} aria-pressed={x.status === s}
                      className="px-2.5 py-1 rounded-full text-[11px] font-semibold"
                      style={{
                        background: x.status === s ? 'var(--c-accent)' : 'var(--c-raised)',
                        color: x.status === s ? '#fff' : 'var(--c-t2)',
                        border: '1px solid var(--c-border)',
                      }}>
                {s ? s.charAt(0) + s.slice(1).toLowerCase() : 'All'}
              </button>
            ))}
          </div>
          {datasetSource && (
            <label className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--c-t2)' }}>
              <input type="checkbox" checked={x.onlyDataset} onChange={e => x.setOnlyDataset(e.target.checked)} />
              Only runs for this dataset
            </label>
          )}
          <input value={x.text} onChange={e => x.setText(e.target.value)}
                 placeholder="Run name, or e.g. metrics.`val_metric/miou` > 0.5"
                 aria-label="Filter runs" title="A run name, or an MLflow filter such as params.lr = '0.001' AND metrics.`val_loss/loss` < 0.5"
                 className="px-3 py-1 rounded-lg text-xs outline-none"
                 style={{ background: 'var(--c-raised)', border: '1px solid var(--c-border)', color: 'var(--c-t1)', width: 300 }} />
          <div className="ml-auto flex items-center gap-2 relative">
            <button className="btn-icon" onClick={() => setPickerOpen(o => !o)} aria-expanded={pickerOpen}
                    style={{ width: 'auto', padding: '0 10px', fontSize: 11 }}>Columns</button>
            <button className="btn-icon" onClick={x.refresh} title="Refresh" aria-label="Refresh"><RefreshCw size={13} /></button>
            {pickerOpen && (
              <div className="absolute right-0 top-full mt-1 p-3 rounded-lg overflow-auto z-20"
                   style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)', width: 280, maxHeight: 340 }}>
                {(['metrics', 'params'] as const).map(kind => (
                  <div key={kind} className="mb-2">
                    <div className="text-[10px] uppercase font-semibold mb-1" style={{ color: 'var(--c-t3)' }}>{kind}</div>
                    {[...new Set([...x.columns[kind], ...seen[kind]])].sort().map(k => (
                      <label key={k} className="flex items-center gap-2 text-xs py-0.5" style={{ color: 'var(--c-t1)' }}>
                        <input type="checkbox" checked={x.columns[kind].includes(k)} onChange={() => toggleColumn(kind, k)} />
                        <span className="truncate" title={k}>{k}</span>
                      </label>
                    ))}
                    {seen[kind].length === 0 && x.columns[kind].length === 0 && (
                      <span className="text-xs" style={{ color: 'var(--c-t3)' }}>None in the current results</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {x.error && (
          <div className="px-4 py-2 text-xs" style={{ color: 'var(--color-danger)' }}>{x.error}</div>
        )}

        {/* Runs table */}
        <div className="flex-1 overflow-auto">
          <table style={{ borderCollapse: 'collapse', width: 'max-content', minWidth: '100%' }}>
            <thead>
              <tr style={{ position: 'sticky', top: 0, zIndex: 5 }}>
                <Th label="Name" sort={x.sort} kind="attribute" id="name" onSort={sortBy} />
                <Th label="Status" sort={x.sort} kind="attribute" id="status" onSort={sortBy} />
                <Th label="Experiment" />
                <Th label="Dataset" />
                <Th label="Preset" />
                <Th label="Origin" />
                <Th label="Started" sort={x.sort} kind="attribute" id="start" onSort={sortBy} />
                <Th label="Duration" />
                <Th label="User" />
                {x.columns.metrics.map(k => <Th key={`m${k}`} label={k} sort={x.sort} kind="metric" id={k} onSort={sortBy} />)}
                {x.columns.params.map(k => <Th key={`p${k}`} label={k} sort={x.sort} kind="param" id={k} onSort={sortBy} />)}
                <Th label="" />
              </tr>
            </thead>
            <tbody>
              {x.runs.map(run => (
                <tr key={run.run_id} onClick={() => onSelectRun(run)} className="cursor-pointer"
                    style={{
                      borderBottom: '1px solid var(--c-border)',
                      background: run.run_id === selectedRunId ? 'var(--c-accent-light)' : undefined,
                    }}>
                  <Td strong>{run.name || run.run_id.slice(0, 8)}</Td>
                  <Td><StatusBadge status={run.status} /></Td>
                  <Td>{experimentNames.get(run.experiment_id) ?? run.experiment_id}</Td>
                  <Td>{run.magnidata['magnidata.dataset_name'] ?? '—'}</Td>
                  <Td>{run.magnidata['magnidata.preset'] ?? '—'}</Td>
                  <Td>{run.magnidata['magnidata.origin'] ?? '—'}</Td>
                  <Td>{run.start_time ? new Date(run.start_time).toLocaleString() : '—'}</Td>
                  <Td>{formatDuration(run)}</Td>
                  <Td>{run.user ?? '—'}</Td>
                  {x.columns.metrics.map(k => <Td key={`m${k}`} mono right>{formatMetric(run.metrics[k])}</Td>)}
                  {x.columns.params.map(k => <Td key={`p${k}`} mono right>{run.params[k] ?? '—'}</Td>)}
                  <Td>
                    {run.mlflow_ui_url && (
                      <a href={run.mlflow_ui_url} target="_blank" rel="noreferrer" onClick={e => e.stopPropagation()}
                         className="flex items-center gap-1 text-[11px]" style={{ color: 'var(--c-accent)' }}>
                        Open in MLflow <ExternalLink size={11} />
                      </a>
                    )}
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
          {x.loading && (
            <div className="flex items-center gap-2 px-4 py-3 text-xs" style={{ color: 'var(--c-t3)' }}>
              <Loader2 size={13} style={{ animation: 'spin 0.7s linear infinite' }} /> Loading runs…
            </div>
          )}
          {!x.loading && !x.error && x.runs.length === 0 && (
            <p className="px-4 py-6 text-sm" style={{ color: 'var(--c-t3)' }}>
              {down ? 'No data while MLflow is unreachable.' : x.selected.length === 0
                ? 'Select an experiment on the left.' : 'No runs match the current filters.'}
            </p>
          )}
          {x.hasMore && !x.loading && (
            <div className="px-4 py-3">
              <button className="btn-primary" onClick={x.loadMore}>Load more</button>
            </div>
          )}
        </div>
      </section>
      {selectedRunId && (
        <RunDetail runId={selectedRunId} knownDatasets={knownDatasets} onLoadDataset={onLoadDataset} onClose={onCloseRun} />
      )}
    </div>
  )
}

interface ThProps {
  label: string
  sort?: SortState
  kind?: ColumnKind
  id?: string
  onSort?: (kind: ColumnKind, key: string) => void
}

function Th({ label, sort, kind, id, onSort }: ThProps) {
  const sortable = !!(onSort && kind && id)
  const active = sortable && sort?.kind === kind && sort?.key === id
  return (
    <th onClick={sortable ? () => onSort!(kind!, id!) : undefined}
        aria-sort={active ? (sort!.dir === 'asc' ? 'ascending' : 'descending') : undefined}
        style={{
          ...thStyle, background: active ? TABLE_HEADER_ACTIVE : TABLE_HEADER,
          color: active ? '#fff' : 'rgba(255,255,255,0.72)', cursor: sortable ? 'pointer' : 'default', userSelect: 'none',
        }}>
      <div className="flex items-center gap-1">
        <span className="truncate" style={{ maxWidth: 160 }} title={label}>{label}</span>
        {sortable && (
          <span style={{ opacity: active ? 1 : 0.3 }}>
            {!active ? <ChevronsUpDown size={10} /> : sort!.dir === 'asc' ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
          </span>
        )}
      </div>
    </th>
  )
}

function Td({ children, mono, right, strong }: { children?: React.ReactNode; mono?: boolean; right?: boolean; strong?: boolean }) {
  return (
    <td style={{
      padding: '6px 10px', whiteSpace: 'nowrap', fontSize: mono ? 11 : 12, color: 'var(--c-t1)',
      fontFamily: mono ? 'JetBrains Mono, monospace' : undefined, textAlign: right ? 'right' : 'left',
      fontWeight: strong ? 600 : undefined, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis',
    }}>
      {children}
    </td>
  )
}

const thStyle: React.CSSProperties = {
  padding: '8px 10px', whiteSpace: 'nowrap', fontSize: 11, fontWeight: 500,
  borderBottom: '1px solid rgba(255,255,255,0.06)', textAlign: 'left', letterSpacing: '.3px',
}
