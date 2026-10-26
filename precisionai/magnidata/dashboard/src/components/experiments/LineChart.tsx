import { useMemo, useState } from 'react'

export interface Series { label: string; color: string; x: number[]; y: number[] }

interface Props {
  title: string
  series: Series[]
  logY?: boolean
  xLabel: string
  height?: number
  /** x value to mark with a vertical line (e.g. the epoch being viewed) */
  marker?: number
}

const W = 360
const M = { l: 44, r: 10, t: 8, b: 26 }

function ticks(lo: number, hi: number, n = 4): number[] {
  if (hi === lo) return [lo]
  return Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n)
}

function fmt(v: number): string {
  if (v === 0) return '0'
  const a = Math.abs(v)
  return a >= 1000 || a < 0.01 ? v.toExponential(1) : Number(v.toPrecision(3)).toString()
}

/** Small SVG line chart: shared axes, one polyline per series, hover readout of the nearest x. */
export function LineChart({ title, series, logY, xLabel, height = 190, marker }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  const plotted = useMemo(
    () => series.map(s => ({ ...s, y: logY ? s.y.map(v => (v > 0 ? Math.log10(v) : NaN)) : s.y })),
    [series, logY],
  )
  const xs = plotted.flatMap(s => s.x)
  const ys = plotted.flatMap(s => s.y).filter(Number.isFinite)
  if (!xs.length || !ys.length) {
    return <ChartFrame title={title}><p className="text-[11px] py-6 text-center" style={{ color: 'var(--c-t3)' }}>No data yet</p></ChartFrame>
  }
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)]
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)]
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5 }
  const sx = (x: number) => M.l + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (W - M.l - M.r)
  const sy = (y: number) => M.t + (1 - (y - y0) / (y1 - y0)) * (height - M.t - M.b)
  const yLabel = (v: number) => fmt(logY ? 10 ** v : v)

  const nearest = hover == null ? null : xs.reduce((best, x) => (Math.abs(x - hover) < Math.abs(best - hover) ? x : best), xs[0])
  return (
    <ChartFrame title={title} legend={plotted}>
      <svg viewBox={`0 0 ${W} ${height}`} width="100%" role="img" aria-label={title}
           onMouseMove={e => {
             const r = e.currentTarget.getBoundingClientRect()
             const px = ((e.clientX - r.left) / r.width) * W
             setHover(x0 + ((px - M.l) / (W - M.l - M.r)) * (x1 - x0))
           }}
           onMouseLeave={() => setHover(null)}>
        {ticks(y0, y1).map(t => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} stroke="var(--c-border)" strokeWidth={1} />
            <text x={M.l - 5} y={sy(t) + 3} fontSize={9} textAnchor="end" fill="var(--c-t3)">{yLabel(t)}</text>
          </g>
        ))}
        {ticks(x0, x1, 3).map(t => (
          <text key={t} x={sx(t)} y={height - 10} fontSize={9} textAnchor="middle" fill="var(--c-t3)">{fmt(t)}</text>
        ))}
        <text x={(W + M.l) / 2} y={height - 0} fontSize={9} textAnchor="middle" fill="var(--c-t3)">{xLabel}</text>
        {plotted.map(s => (
          <polyline key={s.label} fill="none" stroke={s.color} strokeWidth={1.8} strokeLinejoin="round"
                    points={s.x.map((x, i) => (Number.isFinite(s.y[i]) ? `${sx(x)},${sy(s.y[i])}` : null)).filter(Boolean).join(' ')} />
        ))}
        {plotted.map(s => s.x.length === 1 && Number.isFinite(s.y[0]) && (
          <circle key={`dot-${s.label}`} cx={sx(s.x[0])} cy={sy(s.y[0])} r={3} fill={s.color} />
        ))}
        {marker != null && marker >= x0 && marker <= x1 && (
          <line x1={sx(marker)} x2={sx(marker)} y1={M.t} y2={height - M.b} stroke="var(--c-accent)" strokeWidth={1.5} />
        )}
        {nearest != null && <line x1={sx(nearest)} x2={sx(nearest)} y1={M.t} y2={height - M.b} stroke="var(--c-t3)" strokeDasharray="3 3" />}
      </svg>
      {nearest != null && (
        <div className="text-[10px] font-mono flex gap-3 flex-wrap" style={{ color: 'var(--c-t2)' }}>
          <span>{xLabel} {fmt(nearest)}</span>
          {series.map(s => {
            const i = s.x.indexOf(nearest)
            return i >= 0 ? <span key={s.label} style={{ color: s.color }}>{s.label}: {fmt(s.y[i])}</span> : null
          })}
        </div>
      )}
    </ChartFrame>
  )
}

function ChartFrame({ title, legend, children }: { title: string; legend?: Series[]; children: React.ReactNode }) {
  return (
    <div className="rounded-lg p-2.5" style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)' }}>
      <div className="flex items-center justify-between mb-1 gap-2">
        <span className="text-[11px] font-semibold truncate" style={{ color: 'var(--c-t1)' }} title={title}>{title}</span>
        <span className="flex gap-2 shrink-0">
          {legend?.map(s => (
            <span key={s.label} className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--c-t2)' }}>
              <span style={{ width: 10, height: 3, background: s.color, display: 'inline-block', borderRadius: 2 }} />{s.label}
            </span>
          ))}
        </span>
      </div>
      {children}
    </div>
  )
}
