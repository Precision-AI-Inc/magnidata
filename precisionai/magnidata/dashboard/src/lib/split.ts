import type { CSVRow } from '../types'

export const SPLIT_COLUMN = 'split'
/** Largest number of distinct values a column may have to be used as a 3D partition. */
export const MAX_PARTITION_VALUES = 12

/** One fixed colour per split, so train/val look the same in every view. */
export const SPLIT_COLORS: Record<string, string> = {
  train: '#00A0B4',
  val: '#F2A93B',
  test: '#B58AE0',
}
const OTHER_SPLIT_COLOR = '#8A9AA6'

export function splitColor(value: unknown): string {
  return SPLIT_COLORS[String(value)] ?? OTHER_SPLIT_COLOR
}

export function hexToRgb01(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255]
}

/** Rows per split label, in train/val/test order first; rows without a split are not counted. */
export function splitCounts(rows: readonly CSVRow[]): [string, number][] {
  const counts = new Map<string, number>()
  for (const r of rows) {
    const v = r[SPLIT_COLUMN]
    if (v != null && v !== '') counts.set(String(v), (counts.get(String(v)) ?? 0) + 1)
  }
  const order = Object.keys(SPLIT_COLORS)
  return [...counts].sort((a, b) => (order.indexOf(a[0]) + 1 || 99) - (order.indexOf(b[0]) + 1 || 99))
}

/** "train 361 · val 90" */
export function splitSummary(rows: readonly CSVRow[]): string {
  return splitCounts(rows).map(([name, n]) => `${name} ${n.toLocaleString()}`).join(' · ')
}

/** True when the column has between 2 and `max` distinct non-empty values. */
export function partitionEligible(col: string, rows: readonly CSVRow[], max = MAX_PARTITION_VALUES): boolean {
  const seen = new Set<string>()
  for (const r of rows) {
    const v = r[col]
    if (v == null || v === '') continue
    seen.add(String(v))
    if (seen.size > max) return false
  }
  return seen.size >= 2
}

/** Categorical columns usable as a 3D partition; `split` first, then the dataset's own cluster column. */
export function partitionColumns(categorical: readonly string[], rows: readonly CSVRow[], clusterCol?: string | null): string[] {
  const eligible = categorical.filter(c => partitionEligible(c, rows) || c === clusterCol)
  if (clusterCol && !eligible.includes(clusterCol)) eligible.push(clusterCol)
  const rank = (c: string) => (c === SPLIT_COLUMN ? 0 : c === clusterCol ? 1 : 2)
  return eligible.sort((a, b) => rank(a) - rank(b))
}

/** Categorical axes after an overlay: the current ones plus `split` (listed first) when the run provided it. */
export function defaultTrainingCategorical(available: Iterable<string>, current: Iterable<string>, runColumns: readonly string[]): string[] {
  const out = [...current]
  if (new Set(available).has(SPLIT_COLUMN) && runColumns.includes(SPLIT_COLUMN) && !out.includes(SPLIT_COLUMN)) {
    out.unshift(SPLIT_COLUMN)
  }
  return out
}
