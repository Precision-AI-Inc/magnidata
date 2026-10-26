import { useEffect, useMemo, useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import { api } from '../../api'
import type { MetricHistory, RunDetailData } from '../../api'
import { assignRunColors, bestValue, defaultDirection, diffParams } from '../../lib/compare'
import type { Direction } from '../../lib/compare'
import { formatMetric } from '../../lib/experiments'
import { ema, groupMetricKeys, toRelativeTime } from '../../lib/metrics'
import { LineChart } from './LineChart'
import type { Series } from './LineChart'
import { StatusBadge } from './ExperimentsTab'

interface Props {
  runIds: string[]
  knownDatasets: string[]
  /** Overlay the per-image change between the two runs on their dataset. */
  onCompareOnDataset: (a: string, b: string, source: string) => void
  onClose: () => void
}

const DEFAULT_KEYS = ['val_loss/loss', 'val_metric/miou']

export function RunCompare({ runIds, knownDatasets, onCompareOnDataset, onClose }: Props) {
  const [runs, setRuns] = useState<RunDetailData[]>([])
  const [histories, setHistories] = useState<Record<string, Record<string, MetricHistory[]>>>({})
  const [chosen, setChosen] = useState<string[] | null>(null)
  const [directions, setDirections] = useState<Record<string, Direction>>({})
  const [onlyDiff, setOnlyDiff] = useState(true)
  const [alpha, setAlpha] = useState(0)
  const [logY, setLogY] = useState(false)
  const [relTime, setRelTime] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const colors = useMemo(() => assignRunColors(runIds), [runIds])
  const allKeys = useMemo(() => [...new Set(runs.flatMap(r => r.metric_keys))].sort(), [runs])
  const keys = chosen ?? (allKeys.filter(k => DEFAULT_KEYS.includes(k)).length ? allKeys.filter(k => DEFAULT_KEYS.includes(k)) : allKeys.slice(0, 2))
  const keyList = keys.join('\n')

  useEffect(() => {
    let live = true
    Promise.all(runIds.map(id => api.getRun(id)))
      .then(r => { if (live) setRuns(r) })
      .catch(e => { if (live) setError(e instanceof Error ? e.message : String(e)) })
    return () => { live = false }
  }, [runIds])

  useEffect(() => {
    if (!keys.length) return
    let live = true
    api.getAllRunsMetrics(runIds, keys)
      .then(r => { if (live) setHistories(r.runs) })
      .catch(e => { if (live) setError(e instanceof Error ? e.message : String(e)) })
    return () => { live = false }
    // keyList is the stable identity of `keys`
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runIds, keyList])

  const direction = (key: string): Direction => directions[key] ?? defaultDirection(key)
  const runName = (r: RunDetailData) => r.name || r.run_id.slice(0, 8)
  const toggleKey = (k: string) => setChosen(keys.includes(k) ? keys.filter(x => x !== k) : [...keys, k])

  const seriesFor = (key: string): Series[] =>
    runs.flatMap(r => {
      const h = histories[r.run_id]?.[key]
      if (!h?.length) return []
      return [{ label: runName(r), color: colors[r.run_id], x: relTime ? toRelativeTime(h) : h.map(p => p.step), y: ema(h.map(p => p.value), alpha) }]
    })

  const paramRows = diffParams(runs, onlyDiff)

  // Per-image comparison needs exactly two runs with per-sample data. It is drawn on the first run's
  // dataset (the parent, when one run was trained on a curated child of the other's dataset); images
  // only one run saw simply have no change value.
  const known = runs.map(r => r.magnidata['magnidata.dataset_source']).filter(src => src && knownDatasets.includes(src))
  const cmpSource = runs.length === 2 && runs.every(r => r.has_per_sample) ? known[0] : undefined
  const cmpReason = runs.length !== 2 ? 'Select exactly two runs'
    : !runs.every(r => r.has_per_sample) ? 'Both runs need per-sample loss data'
    : !cmpSource ? 'Neither run is linked to a dataset available in this app' : null

  return (
    <section className="flex flex-col flex-1 overflow-auto" style={{ minWidth: 0 }}>
      <div className="flex items-center gap-3 px-4 py-2.5 shrink-0" style={{ borderBottom: '1px solid var(--c-border)' }}>
        <button className="btn-ghost flex items-center gap-1 text-xs px-2 py-1" onClick={onClose}><ArrowLeft size={13} /> Runs</button>
        <h2 className="text-sm font-semibold" style={{ color: 'var(--c-t1)' }}>Comparing {runIds.length} runs</h2>
        <button className="btn-primary" disabled={!!cmpReason} title={cmpReason ?? `Colour ${cmpSource} by how each image changed from the first run to the second`}
                onClick={() => cmpSource && onCompareOnDataset(runs[0].run_id, runs[1].run_id, cmpSource)}
                style={{ fontSize: 11, padding: '4px 10px', opacity: cmpReason ? 0.5 : 1 }}>Compare on dataset</button>
        <div className="flex gap-3 ml-2 flex-wrap">
          {runs.map(r => (
            <span key={r.run_id} className="flex items-center gap-1.5 text-[11px]" style={{ color: 'var(--c-t1)' }}>
              <span style={{ width: 10, height: 10, borderRadius: 2, background: colors[r.run_id], display: 'inline-block' }} />
              {runName(r)} <StatusBadge status={r.status} />
            </span>
          ))}
        </div>
      </div>
      {error && <div className="px-4 py-2 text-xs" style={{ color: 'var(--color-danger)' }}>{error}</div>}

      <div className="p-4 space-y-5">
        {/* Curves */}
        <div>
          <div className="flex items-center gap-3 flex-wrap mb-2">
            <h3 className="text-xs font-semibold" style={{ color: 'var(--c-t1)' }}>Curves</h3>
            <select value={relTime ? 'time' : 'step'} onChange={e => setRelTime(e.target.value === 'time')} aria-label="X axis"
                    className="text-[11px] rounded px-1.5 py-0.5" style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }}>
              <option value="step">x: step</option><option value="time">x: relative time</option>
            </select>
            <label className="text-[11px] flex items-center gap-1.5" style={{ color: 'var(--c-t2)' }}>Smoothing
              <input type="range" min={0} max={0.99} step={0.01} value={alpha} onChange={e => setAlpha(Number(e.target.value))} aria-label="Smoothing" />
              <span className="font-mono w-8">{alpha.toFixed(2)}</span></label>
            <label className="text-[11px] flex items-center gap-1" style={{ color: 'var(--c-t2)' }}>
              <input type="checkbox" checked={logY} onChange={e => setLogY(e.target.checked)} /> Log y</label>
          </div>
          <details className="mb-3 text-xs" style={{ color: 'var(--c-t2)' }}>
            <summary className="cursor-pointer">Metrics ({keys.length} chosen of {allKeys.length})</summary>
            <div className="mt-2 grid gap-x-4" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}>
              {groupMetricKeys(allKeys).flatMap(g => g.keys).map(k => (
                <label key={k} className="flex items-center gap-1.5 py-0.5">
                  <input type="checkbox" checked={keys.includes(k)} onChange={() => toggleKey(k)} />
                  <span className="truncate" title={k}>{k}</span>
                </label>
              ))}
            </div>
          </details>
          <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))' }}>
            {keys.map(k => <LineChart key={k} title={k} series={seriesFor(k)} logY={logY} xLabel={relTime ? 'seconds' : 'step'} />)}
          </div>
        </div>

        {/* Summary */}
        {keys.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--c-t1)' }}>Summary</h3>
            <table style={{ borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <Th>Run</Th>
                  {keys.map(k => (
                    <Th key={k}>
                      <div>{k}</div>
                      <select value={direction(k)} onChange={e => setDirections(prev => ({ ...prev, [k]: e.target.value as Direction }))}
                              aria-label={`Better direction for ${k}`} className="text-[10px] rounded"
                              style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }}>
                        <option value="min">lower is better</option><option value="max">higher is better</option>
                      </select>
                    </Th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {runs.map(r => (
                  <tr key={r.run_id} style={{ borderBottom: '1px solid var(--c-border)' }}>
                    <Td><span style={{ color: colors[r.run_id], fontWeight: 600 }}>{runName(r)}</span></Td>
                    {keys.map(k => {
                      const all = runs.map(x => bestValue((histories[x.run_id]?.[k] ?? []).map(p => p.value), direction(k)))
                      const mine = bestValue((histories[r.run_id]?.[k] ?? []).map(p => p.value), direction(k))
                      const best = bestValue(all.filter((v): v is number => v != null), direction(k))
                      const h = histories[r.run_id]?.[k]
                      const isBest = mine != null && mine === best && runs.length > 1
                      return (
                        <Td key={k} mono highlight={isBest}>
                          <span title="final">{formatMetric(h?.[h.length - 1]?.value)}</span>
                          <span style={{ color: 'var(--c-t3)' }}> · best </span>
                          <span title="best over the run">{formatMetric(mine)}</span>
                        </Td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[10px] mt-1" style={{ color: 'var(--c-t3)' }}>Each cell: final value · best value over the run. Highlighted: best run per metric.</p>
          </div>
        )}

        {/* Params */}
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h3 className="text-xs font-semibold" style={{ color: 'var(--c-t1)' }}>Params</h3>
            <label className="text-[11px] flex items-center gap-1" style={{ color: 'var(--c-t2)' }}>
              <input type="checkbox" checked={onlyDiff} onChange={e => setOnlyDiff(e.target.checked)} /> Only differing</label>
          </div>
          {paramRows.length === 0 ? (
            <p className="text-xs" style={{ color: 'var(--c-t3)' }}>{onlyDiff ? 'No param differs between these runs.' : 'No params.'}</p>
          ) : (
            <table style={{ borderCollapse: 'collapse' }}>
              <thead><tr><Th>Param</Th>{runs.map(r => <Th key={r.run_id}><span style={{ color: colors[r.run_id] }}>{runName(r)}</span></Th>)}</tr></thead>
              <tbody>
                {paramRows.map(row => (
                  <tr key={row.key} style={{ borderBottom: '1px solid var(--c-border)' }}>
                    <Td mono>{row.key}</Td>
                    {row.values.map((v, i) => <Td key={runs[i].run_id} mono highlight={row.differs}>{v ?? '—'}</Td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </section>
  )
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="text-left text-[11px] font-medium px-3 py-1.5" style={{ color: 'var(--c-t2)', borderBottom: '1px solid var(--c-border)' }}>{children}</th>
}

function Td({ children, mono, highlight }: { children: React.ReactNode; mono?: boolean; highlight?: boolean }) {
  return (
    <td className="px-3 py-1.5" style={{
      fontSize: 11.5, fontFamily: mono ? 'JetBrains Mono, monospace' : undefined, color: 'var(--c-t1)',
      background: highlight ? 'var(--c-accent-light)' : undefined, fontWeight: highlight ? 600 : undefined,
    }}>{children}</td>
  )
}
