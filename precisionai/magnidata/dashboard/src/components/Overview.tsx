import type { CSVData, CSVRow, ColMeta } from '../types'

interface Props { data: CSVData; rows: CSVRow[] }

export function Overview({ data, rows }: Props) {
  const total    = data.rows.length
  const filtered = rows.length
  const pct      = total ? ((filtered / total) * 100).toFixed(1) : '0'

  return (
    <div className="overflow-y-auto h-full p-5">
      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-3 mb-5">
        <SummaryCard label="Total rows"      value={total.toLocaleString()}    sub="in file" />
        <SummaryCard label="Filtered rows"   value={filtered.toLocaleString()} sub={`${pct}% of total`} accent />
        <SummaryCard label="Columns"         value={String(data.cols.length)}  sub={`${data.numericCols.length} numeric`} />
        <SummaryCard label="Null values"
          value={data.cols.reduce((s, c) => s + c.nullCount, 0).toLocaleString()}
          sub="across all columns" />
      </div>

      {/* Numeric column stats table */}
      {data.numericCols.length > 0 && (
        <div className="surface mb-5 overflow-hidden">
          <div className="px-4 py-3" style={{ borderBottom: '1px solid var(--c-border)' }}>
            <h3 className="font-semibold text-sm">Numeric Column Statistics</h3>
            <p className="text-xs mt-0.5" style={{ color: 'var(--c-t3)' }}>Computed on all {total.toLocaleString()} rows</p>
          </div>
          <div className="overflow-x-auto">
            <table style={{ borderCollapse: 'collapse', width: '100%' }}>
              <thead>
                <tr style={{ background: 'var(--c-raised)' }}>
                  {['Column', 'Min', 'P25', 'Median', 'Mean', 'P75', 'Max', 'Std', 'Nulls'].map(h => (
                    <th key={h} style={{ padding: '6px 12px', fontSize: 11, fontWeight: 600, textAlign: h === 'Column' ? 'left' : 'right', color: 'var(--c-t2)', borderBottom: '1px solid var(--c-border)', whiteSpace: 'nowrap' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.numericCols.map(col => {
                  const m = data.cols.find(c => c.name === col)!
                  const vals = rows.map(r => r[col] as number).filter(v => v != null && isFinite(v))
                  const live = liveStats(vals)
                  return (
                    <tr key={col} style={{ borderBottom: '1px solid var(--c-border)' }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'var(--c-hover)')}
                        onMouseLeave={e => (e.currentTarget.style.background = '')}>
                      <td style={{ padding: '5px 12px', fontSize: 12, fontWeight: 500, color: 'var(--c-t1)', maxWidth: 160 }}>
                        <span className="truncate block" title={col}>{col}</span>
                      </td>
                      {[live.min, live.p25, live.p50, live.mean, live.p75, live.max, live.std].map((v, i) => (
                        <td key={i} style={{ padding: '5px 12px', textAlign: 'right', fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: 'var(--c-t1)' }}>
                          {fmtN(v)}
                        </td>
                      ))}
                      <td style={{ padding: '5px 12px', textAlign: 'right', fontSize: 11, color: m.nullCount > 0 ? 'var(--c-warn)' : 'var(--c-t3)' }}>
                        {m.nullCount}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Categorical distribution */}
      {data.categoricalCols.length > 0 && (
        <div className="surface overflow-hidden">
          <div className="px-4 py-3" style={{ borderBottom: '1px solid var(--c-border)' }}>
            <h3 className="font-semibold text-sm">Categorical Distributions</h3>
          </div>
          <div className="p-4 grid grid-cols-3 gap-4">
            {data.categoricalCols.map(col => {
              const meta = data.cols.find(c => c.name === col)!
              return <CatDistBar key={col} meta={meta} rows={rows} />
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function SummaryCard({ label, value, sub, accent }:
  { label: string; value: string; sub: string; accent?: boolean }) {
  return (
    <div className="surface p-4 rounded-xl">
      <p className="text-xs mb-1" style={{ color: 'var(--c-t3)' }}>{label}</p>
      <p className="text-2xl font-bold" style={{ color: accent ? 'var(--c-accent)' : 'var(--c-t1)' }}>{value}</p>
      <p className="text-xs mt-0.5" style={{ color: 'var(--c-t3)' }}>{sub}</p>
    </div>
  )
}

function CatDistBar({ meta, rows }: { meta: ColMeta; rows: CSVRow[] }) {
  const freq: Record<string, number> = {}
  rows.forEach(r => {
    const v = String(r[meta.name] ?? '—')
    freq[v] = (freq[v] ?? 0) + 1
  })
  const items = Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 12)
  const max   = items[0]?.[1] ?? 1

  return (
    <div>
      <p className="text-xs font-semibold mb-2" style={{ color: 'var(--c-t2)' }}>{meta.name}</p>
      <div className="flex flex-col gap-1">
        {items.map(([val, count]) => (
          <div key={val} className="flex items-center gap-2">
            <span className="text-[10px] w-16 truncate shrink-0" style={{ color: 'var(--c-t2)' }} title={val}>{val}</span>
            <div className="flex-1 h-3 rounded-full overflow-hidden" style={{ background: 'var(--c-raised)' }}>
              <div className="h-full rounded-full" style={{ width: `${(count / max) * 100}%`, background: 'var(--c-accent)' }} />
            </div>
            <span className="text-[10px] w-8 text-right shrink-0" style={{ color: 'var(--c-t3)' }}>{count}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function liveStats(nums: number[]) {
  if (!nums.length) return { min: NaN, max: NaN, mean: NaN, std: NaN, p25: NaN, p50: NaN, p75: NaN }
  const sorted = [...nums].sort((a, b) => a - b)
  const mean = nums.reduce((s, v) => s + v, 0) / nums.length
  const std  = Math.sqrt(nums.reduce((s, v) => s + (v - mean) ** 2, 0) / nums.length)
  const pct  = (p: number) => { const i = (p / 100) * (sorted.length - 1); const lo = Math.floor(i); return sorted[lo] + (sorted[Math.ceil(i)] - sorted[lo]) * (i - lo) }
  return { min: sorted[0], max: sorted[sorted.length - 1], mean, std, p25: pct(25), p50: pct(50), p75: pct(75) }
}

function fmtN(n: number) {
  if (!isFinite(n)) return '—'
  if (Math.abs(n) >= 10000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(2)
  return n.toPrecision(4).replace(/\.?0+$/, '')
}
