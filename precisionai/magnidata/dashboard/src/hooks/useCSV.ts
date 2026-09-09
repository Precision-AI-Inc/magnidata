import { useState, useCallback } from 'react'
import Papa from 'papaparse'
import type { CSVData, CatEncoding, ColMeta, ColType } from '../types'

function percentile(sorted: number[], p: number): number {
  const i = (p / 100) * (sorted.length - 1)
  const lo = Math.floor(i), hi = Math.ceil(i)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo)
}

function analyzeCol(name: string, rawRows: Record<string, unknown>[]): ColMeta {
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

// ── Camera extraction ──────────────────────────────────────────────────────────
const CROP_WORDS = new Set([
  'corn', 'soybeans', 'soybean', 'generic', 'weed', 'wheat', 'rice',
  'cotton', 'canola', 'sunflower', 'sorghum', 'barley', 'oat',
])

function parseCameraFromStem(stem: string): string {
  const parts = stem.split('-')
  if (parts.length < 4) return 'unknown'
  const hash = parts[parts.length - 1]
  if (!/^[0-9a-f]{8}$/i.test(hash)) return 'unknown'
  // middle = everything between time (index 1) and hash (last)
  const middle = parts.slice(2, -1)
  let camStart = 0
  for (let i = 0; i < middle.length; i++) {
    if (CROP_WORDS.has(middle[i].toLowerCase())) camStart = i + 1
  }
  const camera = middle.slice(camStart).join('-')
  return camera || 'unknown'
}

function buildCatEncodings(
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

const PATH_RE = /^\/|^\w:\\/   // looks like an absolute path
const LFS_POINTER_VERSION = 'version https://git-lfs.github.com/spec/v1'

function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(0)} KB`
  return `${bytes} B`
}

function lfsPointerSize(text: string): number | null {
  const lines = text.split(/\r?\n/)
  if (lines[0]?.trim() !== LFS_POINTER_VERSION) return null
  const sizeLine = lines.find(line => line.startsWith('size '))
  const size = sizeLine ? Number(sizeLine.slice(5)) : NaN
  return Number.isFinite(size) ? size : 0
}

function detectSpecial(cols: ColMeta[], rows: Record<string, unknown>[]) {
  let imagePath: string | null = null
  let maskPath: string | null  = null
  let thumbnailPath: string | null = null
  let stemCol: string | null   = null
  let batchCol: string | null  = null

  for (const c of cols) {
    // Consider any non-numeric column. A path column in a SMALL dataset gets typed
    // 'categorical' (≤30 unique values), so restricting to 'string' would miss it.
    if (c.type === 'numeric') continue
    const lower = c.name.toLowerCase()
    const sample = String(rows[0]?.[c.name] ?? '')

    if (!imagePath && lower.includes('image') && lower.includes('path')) imagePath = c.name
    else if (!imagePath && lower === 'image_path') imagePath = c.name

    if (!maskPath && (lower.includes('mask') || lower.includes('seg')) && lower.includes('path')) maskPath = c.name

    // A per-row pre-generated small preview (e.g. a CDN thumbnail column), distinct
    // from a thumbnail *of the mask* — "thumbnail_image" yes, "thumbnail_masks" no.
    if (!thumbnailPath && lower.includes('thumbnail') && !lower.includes('mask')) thumbnailPath = c.name

    if (!stemCol && (lower === 'stem' || lower === 'name' || lower === 'filename' || lower === 'id')) stemCol = c.name

    if (!batchCol && lower.startsWith('batch')) batchCol = c.name

    // fallback: looks like a path
    if (!imagePath && PATH_RE.test(sample) && lower.includes('path')) imagePath = c.name
  }

  // Second pass: find any path-ish (non-numeric) col as fallback for imagePath
  if (!imagePath) {
    for (const c of cols) {
      if (c.type === 'numeric') continue
      const sample = String(rows[0]?.[c.name] ?? '')
      if (PATH_RE.test(sample)) { imagePath = c.name; break }
    }
  }

  return { imagePath, maskPath, thumbnailPath, stemCol, batchCol }
}

// Predetermined cluster/partition column baked into the dataset, if any. Values may be strings
// (cluster names) or numbers — both are supported.
const CLUSTER_COL_NAMES = new Set([
  'cluster', 'clusters', 'cluster_id', 'clusterid', 'cluster_name', 'cluster_label', 'partition',
])
function detectClusterCol(cols: ColMeta[]): string | null {
  return cols.find(c => CLUSTER_COL_NAMES.has(c.name.trim().toLowerCase()))?.name ?? null
}

export function useCSV() {
  const [data, setData] = useState<CSVData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback((file: File) => {
    setLoading(true)
    setError(null)

    file.text().then(text => {
      const pointerSize = lfsPointerSize(text)
      if (pointerSize !== null) {
        const sizeHint = pointerSize > 0 ? ` (${formatBytes(pointerSize)} expected)` : ''
        setError(`${file.name} is a Git LFS pointer, not the actual CSV${sizeHint}. Run git lfs pull from the repo root, then reload the dashboard.`)
        setLoading(false)
        return
      }

      Papa.parse<Record<string, unknown>>(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete(result) {
          try {
            const rawRows = result.data
            const fields  = result.meta.fields ?? []

            const cols = fields.map(f => analyzeCol(f, rawRows))

            let numericCols     = cols.filter(c => c.type === 'numeric').map(c => c.name)
            let categoricalCols = cols.filter(c => c.type === 'categorical').map(c => c.name)
            let stringCols      = cols.filter(c => c.type === 'string').map(c => c.name)

            const special = detectSpecial(cols, rawRows)
            const clusterCol = detectClusterCol(cols)

            // Derive camera column from stem
            if (special.stemCol) {
              rawRows.forEach(r => {
                r['camera'] = parseCameraFromStem(String(r[special.stemCol!] ?? ''))
              })
              const camMeta = analyzeCol('camera', rawRows)
              cols.push(camMeta)
              if (camMeta.type === 'categorical') categoricalCols.push('camera')
              else stringCols.push('camera')
            }

            // Hide non-feature columns from the table, summaries, and axis pickers:
            //  • the detected image/mask PATH columns (raw paths, not analysable features)
            //  • any column that is entirely empty (every value null/blank)
            // The raw values stay in `rows`, so image previews (via special.imagePath) still work.
            const total = rawRows.length
            const hidden = new Set<string>()
            if (special.imagePath) hidden.add(special.imagePath)
            if (special.maskPath)  hidden.add(special.maskPath)
            if (special.thumbnailPath) hidden.add(special.thumbnailPath)
            for (const c of cols) if (total > 0 && c.nullCount >= total) hidden.add(c.name)
            const visible = (name: string) => !hidden.has(name)

            // Constant categorical columns — a single unique value across all rows (e.g. all
            // "oriented", or a placeholder like "?") — carry no information, so drop them from
            // the categorical feature list. This removes them from the categorical summary,
            // filters, colour-by and categorical axes. (They remain in `cols`/the raw table.)
            const constantCat = new Set(
              cols.filter(c => c.type === 'categorical' && (c.unique ?? 0) <= 1 && c.name !== clusterCol)
                  .map(c => c.name),
            )

            const visibleCols   = cols.filter(c => visible(c.name))
            numericCols     = numericCols.filter(visible)
            categoricalCols = categoricalCols.filter(c => visible(c) && !constantCat.has(c))
            stringCols      = stringCols.filter(visible)

            const catEncodings = buildCatEncodings(categoricalCols, rawRows)
            const rows = rawRows.map((r, i) => ({ ...r, _idx: i }))

            setData({
              rows, cols: visibleCols,
              numericCols, categoricalCols, stringCols,
              catEncodings,
              ...special,
              clusterCol,
              fileName: file.name,
            })
          } catch (e) {
            setError(String(e))
          } finally {
            setLoading(false)
          }
        },
        error(err: Error) {
          setError(err.message)
          setLoading(false)
        },
      })
    }).catch(err => {
      setError(err instanceof Error ? err.message : String(err))
      setLoading(false)
    })
  }, [])

  // Write committed cluster ids into a cluster column (creating it if absent). Only called by
  // "make partition" — exploration / k-means never touches the column. Rows not in `updates`
  // keep their previous value (or null when the column is freshly created).
  const setClusterColumn = useCallback((updates: { idx: number; value: number }[]) => {
    setData(prev => {
      if (!prev) return prev
      const col = prev.clusterCol ?? 'cluster_id'
      const isNew = !prev.cols.some(c => c.name === col)
      const valueByIdx = new Map(updates.map(u => [u.idx, u.value]))
      const rows = prev.rows.map(r => {
        if (valueByIdx.has(r._idx)) return { ...r, [col]: valueByIdx.get(r._idx) }
        return isNew ? { ...r, [col]: null } : r
      })
      // Recompute metadata + type membership for the cluster column only.
      const meta = analyzeCol(col, rows)
      const cols = prev.cols.some(c => c.name === col)
        ? prev.cols.map(c => (c.name === col ? meta : c))
        : [...prev.cols, meta]
      const keep = (list: string[]) => list.filter(c => c !== col)
      const numericCols     = keep(prev.numericCols)
      const categoricalCols = keep(prev.categoricalCols)
      const stringCols      = keep(prev.stringCols)
      if (meta.type === 'numeric')          numericCols.push(col)
      else if (meta.type === 'categorical') categoricalCols.push(col)
      else                                  stringCols.push(col)
      const catEncodings = buildCatEncodings(categoricalCols, rows)
      return { ...prev, rows, cols, numericCols, categoricalCols, stringCols, catEncodings, clusterCol: col }
    })
  }, [])

  // Unload the current dataset (e.g. navigating back to the dataset picker or home).
  const clear = useCallback(() => { setData(null); setError(null) }, [])

  return { data, loading, error, load, setClusterColumn, clear }
}
