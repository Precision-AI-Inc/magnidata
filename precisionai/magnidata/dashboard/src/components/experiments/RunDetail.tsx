import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertTriangle, ChevronDown, ChevronRight, Copy, ExternalLink, File as FileIcon, Folder, X } from 'lucide-react'
import { api } from '../../api'
import type { ArtifactEntry, MetricHistory, RunDetailData, RunMatch } from '../../api'
import { BRAND_CATEGORICAL } from '../../data/vizColors'
import { formatDuration } from '../../lib/experiments'
import { ema, groupMetricKeys, pairTrainVal, toRelativeTime } from '../../lib/metrics'
import { LineChart } from './LineChart'
import type { Series } from './LineChart'
import { StatusBadge } from './ExperimentsTab'

interface Props {
  runId: string
  /** Dataset sources the API knows, to decide whether "Load dataset" is offered. */
  /** Dataset open in the app, if any: the target of "Link to current dataset". */
  currentDataset: string | null
  knownDatasets: string[]
  onLoadDataset: (source: string) => void
  onOverlay: (runId: string, source: string) => void
  onClose: () => void
}

const POLL_MS = 5000
const LIVE = new Set(['SCHEDULED', 'RUNNING'])
const GROUP_LABEL: Record<string, string> = {
  'train_loss/': 'Training loss', 'val_loss/': 'Validation loss', 'val_metric/': 'Validation metrics',
  'Learning_rate/': 'Learning rate', other: 'Other',
}
const TRAIN_COLOR = '#0E7C90'   // deep teal
const VAL_COLOR = '#E08A1E'     // amber: stays distinguishable from the teal train line

type XAxis = 'step' | 'time'

export function RunDetail({ runId, currentDataset, knownDatasets, onLoadDataset, onOverlay, onClose }: Props) {
  const [run, setRun] = useState<RunDetailData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [histories, setHistories] = useState<Record<string, MetricHistory[]>>({})
  const [xAxis, setXAxis] = useState<XAxis>('step')
  const [alpha, setAlpha] = useState(0)
  const [logY, setLogY] = useState(false)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [paramFilter, setParamFilter] = useState('')
  const [linkReport, setLinkReport] = useState<RunMatch | null>(null)
  const [linkError, setLinkError] = useState<string | null>(null)

  const keys = run?.metric_keys ?? []
  const groups = useMemo(() => groupMetricKeys(keys), [keys])
  const visibleKeys = useMemo(() => groups.filter(g => !hidden.has(g.group)).flatMap(g => g.keys), [groups, hidden])
  const status = run?.status

  const loadRun = useCallback(async () => {
    try { setRun(await api.getRun(runId)); setError(null) } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
  }, [runId])

  const loadMetrics = useCallback(async () => {
    const wanted = visibleKeys.slice(0, 20)
    if (!wanted.length) return
    try {
      const fetched = (await api.getRunMetrics(runId, wanted)).metrics
      setHistories(prev => ({ ...prev, ...fetched }))
    } catch { /* the run request surfaces connection errors */ }
  }, [runId, visibleKeys])

  useEffect(() => { setRun(null); setHistories({}); setLinkReport(null); setLinkError(null); void loadRun() }, [loadRun])
  useEffect(() => { void loadMetrics() }, [loadMetrics])
  useEffect(() => {
    if (!status || !LIVE.has(status)) return
    const id = setInterval(() => { void loadRun(); void loadMetrics() }, POLL_MS)
    return () => clearInterval(id)
  }, [status, loadRun, loadMetrics])

  const series = (key: string | undefined, label: string, color: string): Series[] => {
    const h = key ? histories[key] : undefined
    if (!h?.length) return []
    const x = xAxis === 'step' ? h.map(p => p.step) : toRelativeTime(h)
    return [{ label, color, x, y: ema(h.map(p => p.value), alpha) }]
  }
  const xLabel = xAxis === 'step' ? 'step' : 'seconds'

  if (error && !run) {
    return <Panel onClose={onClose}><p className="text-xs p-4" style={{ color: 'var(--color-danger)' }}>{error}</p></Panel>
  }
  if (!run) return <Panel onClose={onClose}><p className="text-xs p-4" style={{ color: 'var(--c-t3)' }}>Loading run…</p></Panel>

  const source = run.magnidata['magnidata.dataset_source']
  const showReport = async () => {
    if (!currentDataset) return
    setLinkError(null)
    try { setLinkReport(await api.getRunMatch(runId, currentDataset)) } catch (e) { setLinkError(e instanceof Error ? e.message : String(e)) }
  }
  const confirmLink = async () => {
    if (!currentDataset) return
    try { await api.linkRun(runId, currentDataset); setLinkReport(null); await loadRun() } catch (e) { setLinkError(e instanceof Error ? e.message : String(e)) }
  }
  const copy = () => { try { void navigator.clipboard.writeText(run.run_id) } catch { /* clipboard unavailable */ } }
  const pairs = pairTrainVal(visibleKeys)
  const tags = Object.entries(run.tags)
  const nd = tags.filter(([k]) => k.startsWith('magnidata.'))
  const rest = tags.filter(([k]) => !k.startsWith('magnidata.') && !k.startsWith('mlflow.'))
  const match = (k: string, v: string) => !paramFilter || `${k} ${v}`.toLowerCase().includes(paramFilter.toLowerCase())

  return (
    <Panel onClose={onClose}>
      <div className="p-4 space-y-4">
        {/* Header */}
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-base font-semibold" style={{ color: 'var(--c-t1)' }}>{run.name || run.run_id.slice(0, 8)}</h2>
            <StatusBadge status={run.status} />
            {run.mlflow_ui_url && (
              <a href={run.mlflow_ui_url} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-[11px]"
                 style={{ color: 'var(--c-accent)' }}>Open in MLflow <ExternalLink size={11} /></a>
            )}
          </div>
          <div className="text-[11px] mt-1 flex gap-3 flex-wrap" style={{ color: 'var(--c-t3)' }}>
            <span>Duration {formatDuration(run)}</span>
            <span>User {run.user ?? '—'}</span>
            <span className="flex items-center gap-1 font-mono">{run.run_id}
              <button onClick={copy} aria-label="Copy run id" title="Copy run id" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', display: 'flex' }}>
                <Copy size={11} /></button></span>
            {source && (
              <span>Dataset <span className="font-mono">{source}</span>
                {knownDatasets.includes(source) && (
                  <button className="ml-2 underline" style={{ color: 'var(--c-accent)' }} onClick={() => onLoadDataset(source)}>Load dataset</button>
                )}
              </span>
            )}
          </div>
          {!source && run.has_per_sample && currentDataset && (
            <div className="mt-2 text-xs" style={{ color: 'var(--c-t2)' }}>
              {!linkReport ? (
                <button className="underline" style={{ color: 'var(--c-accent)' }} onClick={() => void showReport()}>Link to current dataset</button>
              ) : (
                <div className="p-2 rounded-lg" style={{ background: 'var(--c-raised)', border: '1px solid var(--c-border)' }}>
                  {linkReport.matched.toLocaleString()} images match <span className="font-mono">{currentDataset}</span>;
                  {' '}{linkReport.run_only.toLocaleString()} only in the run, {linkReport.dataset_only.toLocaleString()} only in the dataset.
                  <button className="btn-primary ml-3" style={{ fontSize: 11, padding: '2px 10px' }} onClick={() => void confirmLink()}>Link</button>
                  <button className="ml-2 underline" onClick={() => setLinkReport(null)}>Cancel</button>
                </div>
              )}
              {linkError && <span role="alert" style={{ color: 'var(--color-danger)' }}> {linkError}</span>}
            </div>
          )}
          {source && run.has_per_sample && knownDatasets.includes(source) && (
            <button className="mt-2 btn-primary" style={{ fontSize: 11, padding: '3px 10px' }} onClick={() => onOverlay(runId, source)}>Overlay on dataset</button>
          )}
          {run.tags['magnidata.error'] && (
            <div role="alert" className="mt-2 flex items-start gap-2 p-2 rounded-lg text-xs font-semibold"
                 style={{ background: 'var(--color-danger-bg)', color: 'var(--color-danger)', border: '1px solid var(--color-danger)' }}>
              <AlertTriangle size={14} className="shrink-0" /> {run.tags['magnidata.error']}
            </div>
          )}
        </div>

        {/* Metrics */}
        <section>
          <div className="flex items-center gap-3 flex-wrap mb-2">
            <h3 className="text-xs font-semibold" style={{ color: 'var(--c-t1)' }}>Metrics</h3>
            <select value={xAxis} onChange={e => setXAxis(e.target.value as XAxis)} aria-label="X axis" className="text-[11px] rounded px-1.5 py-0.5"
                    style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }}>
              <option value="step">x: step</option><option value="time">x: relative time</option>
            </select>
            <label className="text-[11px] flex items-center gap-1.5" style={{ color: 'var(--c-t2)' }}>Smoothing
              <input type="range" min={0} max={0.99} step={0.01} value={alpha} onChange={e => setAlpha(Number(e.target.value))} aria-label="Smoothing" />
              <span className="font-mono w-8">{alpha.toFixed(2)}</span></label>
            <label className="text-[11px] flex items-center gap-1" style={{ color: 'var(--c-t2)' }}>
              <input type="checkbox" checked={logY} onChange={e => setLogY(e.target.checked)} /> Log y</label>
          </div>
          <div className="flex gap-1.5 flex-wrap mb-3">
            {groups.map(g => (
              <button key={g.group} aria-pressed={!hidden.has(g.group)}
                      onClick={() => setHidden(prev => { const n = new Set(prev); if (n.has(g.group)) n.delete(g.group); else n.add(g.group); return n })}
                      className="px-2 py-0.5 rounded-full text-[10px] font-semibold"
                      style={{ background: hidden.has(g.group) ? 'var(--c-raised)' : 'var(--c-accent-light)', color: hidden.has(g.group) ? 'var(--c-t3)' : 'var(--c-accent)', border: '1px solid var(--c-border)' }}>
                {GROUP_LABEL[g.group]}
              </button>
            ))}
          </div>
          {keys.length === 0 && <p className="text-xs" style={{ color: 'var(--c-t3)' }}>This run has no metrics{LIVE.has(run.status) ? ' yet' : ''}.</p>}
          <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))' }}>
            {pairs.map(p => (
              <LineChart key={p.term} title={p.term === 'loss' ? 'loss (total)' : `loss · ${p.term}`} logY={logY} xLabel={xLabel}
                         series={[...series(p.train, 'train', TRAIN_COLOR), ...series(p.val, 'val', VAL_COLOR)]} />
            ))}
            {visibleKeys.filter(k => !k.startsWith('train_loss/') && !k.startsWith('val_loss/')).map((k, i) => (
              <LineChart key={k} title={k} logY={logY} xLabel={xLabel} series={series(k, k.split('/').pop() ?? k, BRAND_CATEGORICAL[i % BRAND_CATEGORICAL.length])} />
            ))}
          </div>
        </section>

        {/* Params & tags */}
        <section>
          <div className="flex items-center gap-3 mb-2">
            <h3 className="text-xs font-semibold" style={{ color: 'var(--c-t1)' }}>Params &amp; tags</h3>
            <input value={paramFilter} onChange={e => setParamFilter(e.target.value)} placeholder="Search" aria-label="Search params and tags"
                   className="px-2.5 py-1 rounded-lg text-xs outline-none"
                   style={{ background: 'var(--c-raised)', border: '1px solid var(--c-border)', color: 'var(--c-t1)', width: 180 }} />
          </div>
          <KeyValues title="Params" rows={Object.entries(run.params).filter(([k, v]) => match(k, v))} />
          <KeyValues title="MagniData tags" rows={nd.filter(([k, v]) => match(k, v))} />
          <KeyValues title="Other tags" rows={rest.filter(([k, v]) => match(k, v))} />
        </section>

        {/* Artifacts */}
        <section>
          <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--c-t1)' }}>Artifacts</h3>
          <ArtifactTree key={runId} runId={runId} initial={run.artifacts} />
        </section>
      </div>
    </Panel>
  )
}

function Panel({ children, onClose }: { children: React.ReactNode; onClose: () => void }) {
  return (
    <aside className="flex flex-col shrink-0 overflow-auto relative" style={{ width: 'min(760px, 55%)', borderLeft: '1px solid var(--c-border)', background: 'var(--c-bg, transparent)' }}>
      <button onClick={onClose} aria-label="Close run" className="btn-icon absolute right-2 top-2 z-10"><X size={14} /></button>
      {children}
    </aside>
  )
}

function KeyValues({ title, rows }: { title: string; rows: [string, string][] }) {
  if (!rows.length) return null
  return (
    <div className="mb-3">
      <div className="text-[10px] uppercase font-semibold mb-1" style={{ color: 'var(--c-t3)' }}>{title}</div>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <tbody>
          {rows.sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => (
            <tr key={k} style={{ borderBottom: '1px solid var(--c-border)' }}>
              <td className="py-1 pr-3 text-[11px] font-mono align-top" style={{ color: 'var(--c-t2)', width: '40%', wordBreak: 'break-all' }}>{k}</td>
              <td className="py-1 text-[11px] font-mono" style={{ color: 'var(--c-t1)', wordBreak: 'break-all' }}>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const IMAGE_RE = /\.(png|jpe?g|gif|webp)$/i

function ArtifactTree({ runId, initial }: { runId: string; initial: ArtifactEntry[] }) {
  const [children, setChildren] = useState<Record<string, ArtifactEntry[]>>({ '': initial })
  const [open, setOpen] = useState<Set<string>>(new Set())
  const [preview, setPreview] = useState<string | null>(null)

  const toggle = async (path: string) => {
    if (open.has(path)) { setOpen(prev => { const n = new Set(prev); n.delete(path); return n }); return }
    if (!children[path]) {
      try { const r = await api.listRunArtifacts(runId, path); setChildren(prev => ({ ...prev, [path]: r.artifacts })) } catch { return }
    }
    setOpen(prev => new Set(prev).add(path))
  }

  const render = (path: string, depth: number): React.ReactNode =>
    (children[path] ?? []).map(entry => (
      <div key={entry.path}>
        <div className="flex items-center gap-1.5 py-0.5 text-[11px]" style={{ paddingLeft: depth * 14, color: 'var(--c-t1)' }}>
          {entry.is_dir ? (
            <button onClick={() => void toggle(entry.path)} className="flex items-center gap-1" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'inherit' }}>
              {open.has(entry.path) ? <ChevronDown size={12} /> : <ChevronRight size={12} />}<Folder size={12} />{entry.path.split('/').pop()}
            </button>
          ) : (
            <>
              <FileIcon size={12} style={{ marginLeft: 12 }} />
              {IMAGE_RE.test(entry.path)
                ? <button onClick={() => setPreview(entry.path)} className="underline" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--c-accent)' }}>{entry.path.split('/').pop()}</button>
                : <a href={api.runArtifactUrl(runId, entry.path)} download className="underline" style={{ color: 'var(--c-accent)' }}>{entry.path.split('/').pop()}</a>}
              {entry.file_size != null && <span style={{ color: 'var(--c-t3)' }}>{formatSize(entry.file_size)}</span>}
            </>
          )}
        </div>
        {entry.is_dir && open.has(entry.path) && render(entry.path, depth + 1)}
      </div>
    ))

  if (!children[''].length) return <p className="text-xs" style={{ color: 'var(--c-t3)' }}>No artifacts.</p>
  return (
    <div>
      {render('', 0)}
      {preview && (
        <div className="mt-2">
          <div className="text-[10px] flex items-center gap-2" style={{ color: 'var(--c-t3)' }}>{preview}
            <button onClick={() => setPreview(null)} aria-label="Close preview"><X size={11} /></button></div>
          <img src={api.runArtifactUrl(runId, preview)} alt={preview} style={{ maxWidth: '100%', maxHeight: 320, imageRendering: 'pixelated' }} />
        </div>
      )}
    </div>
  )
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
