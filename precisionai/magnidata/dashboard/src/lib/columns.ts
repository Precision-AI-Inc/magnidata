import type { CatEncoding, ColMeta, ColType } from '../types'

function percentile(sorted: number[], p: number): number {
  const i = (p / 100) * (sorted.length - 1)
  const lo = Math.floor(i), hi = Math.ceil(i)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo)
}

export function analyzeCol(name: string, rawRows: Record<string, unknown>[]): ColMeta {
  const values = rawRows.map(r => r[name])
  const nullCount = values.filter(v => v === null || v === undefined || v === '').length

  const nums = values.filter(v => typeof v === 'number' && isFinite(v as number)) as number[]
  const nonNullStrs = values.filter(v => v != null && v !== '') as unknown[]

  // Numeric if ≥70% of all rows are numbers, OR every non-null value is a number
  // (the second condition handles sparse columns like gsd_abs that have many nulls)
  const allNonNullAreNumeric = nonNullStrs.length > 0 && nums.length === nonNullStrs.length
  if (nums.length >= rawRows.length * 0.7 || allNonNullAreNumeric) {
    // Numeric column
    const sorted = [...nums].sort((a, b) => a - b)
    const mean = nums.reduce((s, v) => s + v, 0) / nums.length
    const std  = Math.sqrt(nums.reduce((s, v) => s + (v - mean) ** 2, 0) / nums.length)
    return {
      name, type: 'numeric', nullCount,
      min: sorted[0], max: sorted[sorted.length - 1],
      mean, std,
      p25: percentile(sorted, 25),
      p50: percentile(sorted, 50),
      p75: percentile(sorted, 75),
    }
  }

  // String / categorical
  const freq: Record<string, number> = {}
  nonNullStrs.forEach(v => { const s = String(v); freq[s] = (freq[s] ?? 0) + 1 })
  const unique = Object.keys(freq).length
  const topValues = Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20) as [string, number][]

  const type: ColType = unique <= 30 ? 'categorical' : 'string'
  return { name, type, nullCount, unique, topValues }
}

export function buildCatEncodings(
  catCols: string[],
  rows: Record<string, unknown>[],
): Record<string, CatEncoding> {
  const result: Record<string, CatEncoding> = {}
  for (const col of catCols) {
    const freq: Record<string, number> = {}
    rows.forEach(r => { const v = String(r[col] ?? ''); freq[v] = (freq[v] ?? 0) + 1 })
    const sorted = Object.entries(freq).sort((a, b) => b[1] - a[1]).map(([v]) => v)
    result[col] = { sorted, encode: Object.fromEntries(sorted.map((v, i) => [v, i])) }
  }
  return result
}
