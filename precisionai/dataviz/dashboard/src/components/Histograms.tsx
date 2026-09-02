import { useEffect, useRef, useState } from 'react'
import type { CSVData, CSVRow } from '../types'

interface Props { data: CSVData; rows: CSVRow[] }

const COLS_PER_ROW = 4

export function Histograms({ data, rows }: Props) {
  const [page, setPage] = useState(0)
  const numCols  = data.numericCols
  const catCols  = data.categoricalCols
  const perPage  = COLS_PER_ROW * 3
  const totalPages = Math.ceil(numCols.length / perPage)
  const visibleNum = numCols.slice(page * perPage, (page + 1) * perPage)

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4">
        {/* Numeric histograms */}
        {visibleNum.length > 0 && (
          <>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold" style={{ color: 'var(--c-t1)' }}>
                Numeric Distributions
                <span className="ml-2 text-xs font-normal" style={{ color: 'var(--c-t3)' }}>
                  ({rows.length.toLocaleString()} rows · {numCols.length} columns)
                </span>
              </h3>
              {totalPages > 1 && (
                <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--c-t2)' }}>
                  <button className="btn-ghost py-1 px-2 text-xs" disabled={page === 0}
                          onClick={() => setPage(p => p - 1)}>← Prev</button>
                  <span>{page + 1} / {totalPages}</span>
                  <button className="btn-ghost py-1 px-2 text-xs" disabled={page === totalPages - 1}
                          onClick={() => setPage(p => p + 1)}>Next →</button>
                </div>
              )}
            </div>
            <div className={`grid gap-3 mb-5`}
                 style={{ gridTemplateColumns: `repeat(${COLS_PER_ROW}, 1fr)` }}>
              {visibleNum.map(col => (
                <HistogramCard key={col} col={col} rows={rows} />
              ))}
            </div>
          </>
        )}

        {/* Categorical bar charts */}
        {catCols.length > 0 && (
          <>
            <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--c-t1)' }}>
              Categorical Distributions
            </h3>
            <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${COLS_PER_ROW}, 1fr)` }}>
              {catCols.map(col => (
                <CatBarCard key={col} col={col} rows={rows} />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ── Histogram (numeric) ────────────────────────────────────────────────────────
function HistogramCard({ col, rows }: { col: string; rows: CSVRow[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const vals = rows.map(r => r[col] as number).filter(v => v != null && isFinite(v))

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !vals.length) return
    const ctx = canvas.getContext('2d')!

    const W = canvas.offsetWidth || 200
    const H = 80
    canvas.width  = W * window.devicePixelRatio
    canvas.height = H * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    const BINS = 24
    const mn = Math.min(...vals), mx = Math.max(...vals)
    const range = mx - mn || 1
    const bins  = new Array(BINS).fill(0)
    vals.forEach(v => { const i = Math.min(BINS - 1, Math.floor(((v - mn) / range) * BINS)); bins[i]++ })
    const maxBin = Math.max(...bins)

    const barW = W / BINS
    const pad  = 4

    ctx.clearRect(0, 0, W, H)
    bins.forEach((count, i) => {
      const bh = ((count / maxBin) * (H - pad)) || 0
      const x  = i * barW
      const y  = H - bh
      ctx.fillStyle = 'rgba(0,106,124,0.75)'
      ctx.beginPath()
      ctx.roundRect(x + 1, y, barW - 2, bh, [2, 2, 0, 0])
      ctx.fill()
    })
  }, [vals])

  const mean = vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 0
  const min  = vals.length ? Math.min(...vals) : 0
  const max  = vals.length ? Math.max(...vals) : 0

  return (
    <div className="surface p-3 rounded-xl">
      <p className="text-[11px] font-medium mb-1 truncate" style={{ color: 'var(--c-t1)' }} title={col}>{col}</p>
      <canvas ref={canvasRef} style={{ width: '100%', height: 80, display: 'block' }} />
      <div className="flex justify-between mt-1.5">
        <Micro label="min"  value={fmtN(min)} />
        <Micro label="mean" value={fmtN(mean)} accent />
        <Micro label="max"  value={fmtN(max)} />
      </div>
    </div>
  )
}

// ── Categorical bar chart ──────────────────────────────────────────────────────
function CatBarCard({ col, rows }: { col: string; rows: CSVRow[] }) {
  const freq: Record<string, number> = {}
  rows.forEach(r => { const v = String(r[col] ?? '—'); freq[v] = (freq[v] ?? 0) + 1 })
  const items = Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 10)
  const max   = items[0]?.[1] ?? 1

  return (
    <div className="surface p-3 rounded-xl">
      <p className="text-[11px] font-medium mb-2 truncate" style={{ color: 'var(--c-t1)' }} title={col}>{col}</p>
      <div className="flex flex-col gap-1">
        {items.map(([val, count]) => (
          <div key={val} className="flex items-center gap-1.5">
            <span className="text-[10px] w-12 truncate shrink-0" title={val} style={{ color: 'var(--c-t2)' }}>{val}</span>
            <div className="flex-1 h-2.5 rounded-full overflow-hidden" style={{ background: 'var(--c-raised)' }}>
              <div className="h-full rounded-full" style={{ width: `${(count / max) * 100}%`, background: 'var(--c-accent)' }} />
            </div>
            <span className="text-[10px] w-6 text-right shrink-0" style={{ color: 'var(--c-t3)' }}>{count}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function Micro({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="text-center">
      <p className="text-[9px] uppercase tracking-wide" style={{ color: 'var(--c-t3)' }}>{label}</p>
      <p className="text-[10px] font-mono font-medium" style={{ color: accent ? 'var(--c-accent)' : 'var(--c-t2)' }}>{value}</p>
    </div>
  )
}

function fmtN(n: number) {
  if (!isFinite(n)) return '—'
  if (Math.abs(n) >= 10000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(2)
  return n.toPrecision(3).replace(/\.?0+$/, '')
}
