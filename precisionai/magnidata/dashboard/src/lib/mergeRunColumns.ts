import type { CSVData, CSVRow } from '../types'
import { analyzeCol, buildCatEncodings } from './columns'

export interface MergeResult {
  data: CSVData
  added: string[]
  matched: number
  unmatched: number
}

/** Same join rule as the API: the basename of an image path. */
export function imageKey(value: unknown): string {
  return String(value ?? '').split(/[\\/]/).pop() ?? ''
}

const CATEGORICAL_RUN_COLUMNS = new Set(['split'])

/** Remove the columns a previous overlay added; returns `data` untouched when there are none. */
export function removeRunColumns(data: CSVData): CSVData {
  const previous = data.runColumns
  if (!previous?.length) return data
  const drop = new Set(previous)
  const rows = data.rows.map(r => {
    const copy = { ...r }
    for (const c of drop) delete copy[c]
    return copy
  })
  const keep = (list: string[]) => list.filter(c => !drop.has(c))
  const categoricalCols = keep(data.categoricalCols)
  return {
    ...data,
    rows,
    cols: data.cols.filter(c => !drop.has(c.name)),
    numericCols: keep(data.numericCols),
    categoricalCols,
    catEncodings: buildCatEncodings(categoricalCols, rows),
    runColumns: undefined,
  }
}

/**
 * Join a run's derived per-sample columns onto the dataset by image basename.
 *
 * Columns from an earlier overlay are replaced, not duplicated. Rows without training data get
 * null. `split` is categorical, everything else numeric.
 *
 * @param runRows parsed rows of the run's columns CSV; each has `image_name`
 * @param imageKeyCol dataset column holding the image path (usually `data.imagePath`)
 */
export function mergeRunColumns(
  data: CSVData, runRows: Record<string, unknown>[], imageKeyCol: string | null, prefix = '',
): MergeResult {
  const base = removeRunColumns(data)
  if (!imageKeyCol || !runRows.length) return { data: base, added: [], matched: 0, unmatched: base.rows.length }

  const names = Object.keys(runRows[0]).filter(k => k !== 'image_name')
  const added = names.map(n => prefix + n)
  const byImage = new Map(runRows.map(r => [imageKey(r.image_name), r]))

  let matched = 0
  const rows: CSVRow[] = base.rows.map(row => {
    const hit = byImage.get(imageKey(row[imageKeyCol]))
    if (hit) matched++
    const next: CSVRow = { ...row }
    names.forEach((n, i) => { next[added[i]] = hit ? (hit[n] ?? null) : null })
    return next
  })

  const metas = added.map((col, i) => {
    const meta = analyzeCol(col, rows)
    return CATEGORICAL_RUN_COLUMNS.has(names[i]) ? { ...meta, type: 'categorical' as const } : meta
  })
  const numericCols = [...base.numericCols]
  const categoricalCols = [...base.categoricalCols]
  metas.forEach(m => (m.type === 'categorical' ? categoricalCols : numericCols).push(m.name))
  // A column that is entirely empty (the run matched none of these rows) is not an axis.
  const usable = new Set(metas.filter(m => m.nullCount < rows.length).map(m => m.name))
  const merged: CSVData = {
    ...base,
    rows,
    cols: [...base.cols, ...metas],
    numericCols: numericCols.filter(c => !added.includes(c) || usable.has(c)),
    categoricalCols: categoricalCols.filter(c => !added.includes(c) || usable.has(c)),
    catEncodings: {},
    runColumns: added,
  }
  merged.catEncodings = buildCatEncodings(merged.categoricalCols, rows)
  return { data: merged, added, matched, unmatched: rows.length - matched }
}
