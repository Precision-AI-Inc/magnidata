import { useEffect, useRef, useState, useMemo, useCallback, type CSSProperties } from 'react'
import type Plotly from 'plotly.js-dist-min'
import { Play, ChevronUp, ChevronDown, ChevronsUpDown, Download, Columns, X, RotateCcw } from 'lucide-react'
import type { CSVData, CSVRow } from '../types'
import type { Partitions } from '../hooks/usePartitions'
import { ColTooltip } from './ColTooltip'
import { Graph3D } from './Graph3D'
import { BRAND_CATEGORICAL, BRAND_SEQUENTIAL, TABLE_HEADER, TABLE_HEADER_ACTIVE } from '../data/vizColors'

interface Props {
  data: CSVData
  rows: CSVRow[]                                           // all rows (data.rows)
  embeddingsAvailable?: boolean                           // whether the API has embeddings for this dataset
  datasetSource?: string | null                           // dataset CSV path (for server-side compute)
  onConstraintsChange: (c: Record<string, [number, number][]>) => void
  onPlay: (row: CSVRow, rows: CSVRow[]) => void            // rows = set to navigate in the preview
  partitions?: Partitions                                 // persistent cluster_id annotations
  onCommitClusterColumn?: (updates: { idx: number; value: number }[]) => void  // write cluster col on commit
  removed?: Set<number>                                   // _idx of images staged for removal (excluded from selection)
  // Plot controls — lifted to App so they render in the tab bar.
  colorBy: string
  setColorBy: (v: string) => void
  sortMode: 'default' | 'variance' | 'corr'
  setSortMode: (v: 'default' | 'variance' | 'corr') => void
  showFields: boolean
  setShowFields: (v: boolean | ((p: boolean) => boolean)) => void
}

const PALETTE = BRAND_CATEGORICAL
const PC_HEIGHT = 330

// Brand teal → emerald → mint sequential scale (shared with Graph3D via vizColors).
// Passed as an explicit array (not a named string) so it renders correctly on the very first
// Plotly.react draw and matches the 3D view exactly.
const WARM_SCALE: [number, string][] = BRAND_SEQUENTIAL


// Axes selected by default in the parallel-coordinates view (filtered to those present
// in the loaded dataset). Anything not listed here starts unchecked in the Fields panel.
const DEFAULT_NUMERIC_COLS = [
  'complexity_score', 'width', 'annotation_ratio', 'fg_green_mean',
  'mean_pairwise_color_dist', 'class_entropy', 'nima_ava', 'niqe', 'brisque', 'gsd',
]
const FALLBACK_NUMERIC_COLS = [
  'complexity_score', 'category', 'colorfulness', 'luma_entropy',
  'blur_laplacian', 'tenengrad', 'noise_sigma', 'rms_contrast', 'dynamic_range',
  'shadow_edge_ratio', 'overexpose_ratio', 'underexpose_ratio', 'wb_rb_ratio',
  'wb_r_gain', 'wb_b_gain', 'height', 'width',
]
const DEFAULT_CATEGORICAL_COLS = ['camera', 'cluster', 'cluster_l2', 'camera_angle']
const MIN_DEFAULT_AXES = 5
const MAX_DEFAULT_AXES = 12

function hasNumericVariation(rows: CSVRow[], col: string): boolean {
  let first: number | null = null
  for (const r of rows) {
    const v = r[col] as number
    if (v == null || !isFinite(v)) continue
    if (first == null) { first = v; continue }
    if (v !== first) return true
  }
  return false
}

function defaultNumericSelection(numericCols: string[], rows: CSVRow[]): Set<string> {
  const available = new Set(numericCols)
  const chosen: string[] = []
  const seen = new Set<string>()
  const add = (col: string, allowConstant = false) => {
    if (!available.has(col) || seen.has(col)) return
    if (!allowConstant && !hasNumericVariation(rows, col)) return
    seen.add(col)
    chosen.push(col)
  }

  DEFAULT_NUMERIC_COLS.forEach(c => add(c))
  if (chosen.length < MIN_DEFAULT_AXES) FALLBACK_NUMERIC_COLS.forEach(c => add(c))
  if (chosen.length < MIN_DEFAULT_AXES) numericCols.forEach(c => add(c))
  if (!chosen.length) numericCols.forEach(c => add(c, true))
  return new Set(chosen.slice(0, MAX_DEFAULT_AXES))
}

function defaultCategoricalSelection(categoricalCols: string[]): Set<string> {
  const available = new Set(categoricalCols)
  return new Set(DEFAULT_CATEGORICAL_COLS.filter(c => available.has(c)).slice(0, 2))
}

// Deep-equality for the brush constraint map — used to skip redundant state updates (and the
// downstream Graph3D / table recompute) when a restyle reports the same selection.
type ConstraintMap = Record<string, [number, number][]>
function sameConstraints(a: ConstraintMap, b: ConstraintMap): boolean {
  const ka = Object.keys(a), kb = Object.keys(b)
  if (ka.length !== kb.length) return false
  for (const k of ka) {
    const ra = a[k], rb = b[k]
    if (!rb || ra.length !== rb.length) return false
    for (let i = 0; i < ra.length; i++) {
      if (ra[i][0] !== rb[i][0] || ra[i][1] !== rb[i][1]) return false
    }
  }
  return true
}

export function ParallelCoords({ data, rows, embeddingsAvailable, datasetSource, onConstraintsChange, onPlay, partitions, onCommitClusterColumn, removed, colorBy, setColorBy, sortMode, setSortMode, showFields, setShowFields }: Props) {
  const divRef  = useRef<HTMLDivElement>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const plotRef = useRef<typeof Plotly | null>(null)

  // Height of the parallel-coordinates panel (top row); the bottom row (3D + table) flexes to fill.
  // Draggable via the splitter handle between the two rows.
  const [pcHeight, setPcHeight] = useState(PC_HEIGHT)

  const startResize = (e: React.MouseEvent) => {
    e.preventDefault()
    const startY = e.clientY
    const startH = pcHeight
    const onMove = (ev: MouseEvent) => {
      const max = (rootRef.current?.clientHeight ?? 800) - 300   // leave room for the bottom row (min 280)
      setPcHeight(Math.max(160, Math.min(max, startH + (ev.clientY - startY))))
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.body.style.cursor = 'row-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  // colorBy / sortMode / showFields are controlled by App (rendered in the tab bar).
  // App seeds colorBy via an effect (async), so on the very first paint it can be ''. Use a
  // synchronous fallback for drawing so the plot is coloured immediately, and sync it back up.
  const colorByEff = colorBy || (data.numericCols.includes('complexity_score')
    ? 'complexity_score'
    : (data.categoricalCols[0] ?? data.numericCols[0] ?? ''))
  useEffect(() => { if (!colorBy && colorByEff) setColorBy(colorByEff) }, [colorBy, colorByEff, setColorBy])
  const [constraints,  setConstraints]  = useState<Record<string, [number, number][]>>({})
  const [fieldSearch,     setFieldSearch]     = useState('')
  const [selectedCols,    setSelectedCols]    = useState<Set<string>>(
    () => defaultNumericSelection(data.numericCols, rows)
  )
  // Categorical axes — default to the most useful low-cardinality context columns.
  const [selectedCatCols, setSelectedCatCols] = useState<Set<string>>(
    () => defaultCategoricalSelection(data.categoricalCols)
  )

  useEffect(() => {
    setSelectedCols(defaultNumericSelection(data.numericCols, rows))
    setSelectedCatCols(defaultCategoricalSelection(data.categoricalCols))
    setConstraints({})
    onConstraintsChange({})
  }, [data.fileName]) // eslint-disable-line

  // 3D cluster selection (null = no selection; Set = row indices selected from 3D view)
  const [filter3D, setFilter3D] = useState<Set<number> | null>(null)
  useEffect(() => setFilter3D(null), [rows])

  // ── Brush-capture plumbing (kept in refs so the once-attached restyle listener stays correct) ──
  const restyleTimer       = useRef<number | null>(null)  // single cancellable debounce timer
  const programmaticRedraw = useRef(false)                // true around Plotly.react → ignore self-restyles
  const attachedEl         = useRef<HTMLElement | null>(null) // element the restyle listener is bound to
  const extractRef         = useRef<() => void>(() => {}) // latest extractConstraints (stable wrapper calls this)
  const onConstraintsChangeRef = useRef(onConstraintsChange)
  useEffect(() => { onConstraintsChangeRef.current = onConstraintsChange }, [onConstraintsChange])
  const constraintsRef = useRef(constraints)
  useEffect(() => { constraintsRef.current = constraints }, [constraints])
  // Clear any pending debounce on unmount
  useEffect(() => () => { if (restyleTimer.current != null) window.clearTimeout(restyleTimer.current) }, [])

  // Active axes = selected ∩ numericCols (preserves original order)
  const activeCols = useMemo(
    () => data.numericCols.filter(c => selectedCols.has(c)),
    [data.numericCols, selectedCols],
  )

  // Active categorical axes (appended after numeric on the chart)
  const activeCatCols = useMemo(
    () => data.categoricalCols.filter(c => selectedCatCols.has(c)),
    [data.categoricalCols, selectedCatCols],
  )

  // Rows passing PC brush AND 3D cluster selection (intersection)
  const filteredRows = useMemo<CSVRow[]>(() => {
    const entries = Object.entries(constraints).filter(([col, r]) => {
      if (!r.length) return false
      return selectedCols.has(col) || selectedCatCols.has(col)
    })
    let result = entries.length ? rows.filter(row => {
      for (const [col, ranges] of entries) {
        const enc = data.catEncodings[col]
        if (enc) {
          const encoded = enc.encode[String(row[col] ?? '')]
          if (encoded == null || !ranges.some(([lo, hi]) => encoded >= lo && encoded <= hi)) return false
        } else {
          const raw = row[col] as number
          // null/missing value never satisfies a range constraint.
          if (raw == null || !isFinite(raw)) return false
          if (!ranges.some(([lo, hi]) => raw >= lo && raw <= hi)) return false
        }
      }
      return true
    }) : rows
    if (filter3D) result = result.filter(r => filter3D.has(r._idx as number))
    // Staged removals are excluded from the selection — dimmed in 3D, gone from the table/export.
    if (removed && removed.size) result = result.filter(r => !removed.has(r._idx as number))
    return result
  }, [rows, constraints, selectedCols, selectedCatCols, filter3D, removed])

  // ── Color helpers ──────────────────────────────────────────────────────────
  // displayRows: the rows to assign colors to; cmin/cmax always derived from all rows for stable scale
  function buildColorData(col: string, displayRows: CSVRow[]) {
    const isCat = data.categoricalCols.includes(col)
    if (isCat) {
      const enc = data.catEncodings[col]
      if (enc) {
        const n         = enc.sorted.length
        const colorscale: [number, string][] = enc.sorted.map((_, i) =>
          [n > 1 ? i / (n - 1) : 0, PALETTE[i % PALETTE.length]]
        )
        return {
          colorData: displayRows.map(r => enc.encode[String(r[col] ?? '')] ?? 0),
          colorscale,
          cmin: 0, cmax: Math.max(1, n - 1),
          tickvals: enc.sorted.map((_, i) => i),
          ticktext:  enc.sorted,
        }
      }
    }
    // Numeric: use 2nd–98th percentile for cmin/cmax so outliers don't push all values to one end of the scale
    const nonNullVals: number[] = []
    for (const r of rows) {
      const v = r[col] as number
      if (v != null && isFinite(v)) nonNullVals.push(v)
    }
    let cmin: number, cmax: number
    if (nonNullVals.length === 0) {
      cmin = 0; cmax = 1
    } else {
      nonNullVals.sort((a, b) => a - b)
      const lo = Math.max(0, Math.round(0.02 * (nonNullVals.length - 1)))
      const hi = Math.min(nonNullVals.length - 1, Math.round(0.98 * (nonNullVals.length - 1)))
      cmin = nonNullVals[lo]
      cmax = nonNullVals[hi]
      if (cmin >= cmax) { cmin = nonNullVals[0]; cmax = nonNullVals[nonNullVals.length - 1] }
    }
    return {
      colorData: displayRows.map(r => {
        const v = r[col] as number
        return v != null && isFinite(v) ? v : cmin
      }),
      colorscale: WARM_SCALE,
      cmin, cmax,
      tickvals: undefined, ticktext: undefined,
    }
  }

  // ── Axis sort ──────────────────────────────────────────────────────────────
  function getSortedCols() {
    if (sortMode === 'default') {
      if (activeCols.includes('complexity_score')) {
        return ['complexity_score', ...activeCols.filter(c => c !== 'complexity_score')]
      }
      return activeCols
    }
    const refCol  = colorByEff && activeCols.includes(colorByEff) ? colorByEff : activeCols[0]
    if (!refCol) return activeCols
    const refVals = rows.map(r => r[refCol] as number ?? 0)
    const refMu   = refVals.reduce((s, v) => s + v, 0) / refVals.length
    const refStd  = Math.sqrt(refVals.reduce((s, v) => s + (v - refMu) ** 2, 0) / refVals.length)
    const stats   = Object.fromEntries(activeCols.map(col => {
      const vals = rows.map(r => r[col] as number ?? 0)
      const mu   = vals.reduce((s, v) => s + v, 0) / vals.length
      const std  = Math.sqrt(vals.reduce((s, v) => s + (v - mu) ** 2, 0) / vals.length)
      const cov  = vals.reduce((s, v, i) => s + (v - mu) * (refVals[i] - refMu), 0) / vals.length
      return [col, { variance: std ** 2, corr: std > 0 && refStd > 0 ? Math.abs(cov / (std * refStd)) : 0 }]
    }))
    return [...activeCols].sort((a, b) =>
      sortMode === 'variance' ? stats[b].variance - stats[a].variance : stats[b].corr - stats[a].corr
    )
  }

  // ── Plotly draw ────────────────────────────────────────────────────────────
  async function drawPlot() {
    if (!divRef.current || (!activeCols.length && !activeCatCols.length)) return
    if (!plotRef.current) plotRef.current = (await import('plotly.js-dist-min')).default
    const Plotly = plotRef.current

    const sortedCols  = getSortedCols()
    // When a 3D cluster is selected show only those rows; axis ranges stay pinned to full data
    // Drop removed rows from the drawn lines (axis ranges below still use full `rows` so they're stable).
    const baseRows = (removed && removed.size) ? rows.filter(r => !removed.has(r._idx as number)) : rows
    const displayRows = filter3D ? filteredRows : baseRows
    const { colorData, colorscale, cmin, cmax, tickvals, ticktext } = buildColorData(colorByEff, displayRows)

    // Re-apply any active brush so the selection survives a redraw (color-by / sort / axis change).
    const crProp = (col: string) => {
      const cr = constraints[col]
      if (!cr || !cr.length) return {}
      return { constraintrange: (cr.length === 1 ? cr[0] : cr) as unknown }
    }

    // Numeric dimensions (sorted by user preference). Always pin a slightly PADDED full-data range
    // (computed from all rows so it's stable across redraws and 3D selections), scaled linearly from
    // the data's actual min to its actual max. Without padding, Plotly auto-ranges the axis exactly to
    // [min, max]; a brush dragged to the data extreme then lands on the axis boundary, which Plotly
    // interprets as "no constraint" and clears the selection. The padding keeps the data inside the
    // axis so brushing to the very top/bottom stays a valid sub-range.
    const numericDims = sortedCols.map(col => {
      const values = displayRows.map(r => (r[col] as number) ?? 0)
      let lo = Infinity, hi = -Infinity
      for (const r of rows) {
        const v = (r[col] as number) ?? 0
        if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v }
      }
      if (!isFinite(lo) || !isFinite(hi)) return { label: col, values, ...crProp(col) }
      const pad = hi > lo ? (hi - lo) * 0.03 : (Math.abs(hi) * 0.03 || 1)
      return { label: col, values, range: [lo - pad, hi + pad], ...crProp(col) }
    })

    // Categorical dimensions — encoded as integers with tick labels.
    // Pad the range by half a cell beyond the first/last category so a brush snapped to a category
    // cell ([i-0.5, i+0.5]) never coincides with the axis boundary — Plotly clears any constraint that
    // touches the edge, which would otherwise make the brush disappear when selecting the first/last
    // category (and then erase the whole query).
    const catDims = activeCatCols.flatMap(col => {
      const enc = data.catEncodings[col]
      if (!enc) return []
      return [{
        label: col,
        values: displayRows.map(r => enc.encode[String(r[col] ?? '')] ?? -1),
        tickvals: enc.sorted.map((_, i) => i),
        ticktext: enc.sorted,
        range: [-1, enc.sorted.length],
        ...crProp(col),
      }]
    })

    const trace = {
      type: 'parcoords' as const,
      line: {
        color: colorData as number[],
        colorscale: colorscale as Plotly.ColorScale,
        cmin, cmax, showscale: true,
        colorbar: {
          title: { text: colorByEff, font: { size: 10, color: 'rgba(206,238,228,0.88)' } as Partial<Plotly.Font> },
          thickness: 12, len: 0.65,
          tickfont: { size: 9, color: 'rgba(188,224,214,0.75)' } as Partial<Plotly.Font>,
          outlinecolor: 'rgba(90,200,185,0.28)',
          ...(tickvals ? { tickvals, ticktext } : {}),
        } as Partial<Plotly.ColorBar>,
      } as Partial<Plotly.PlotData>['line'],
      dimensions: [...numericDims, ...catDims] as unknown[],
      labelangle: -30,
      labelfont: { size: 9, color: 'rgba(206,238,228,0.92)' } as Partial<Plotly.Font>,
      tickfont:  { size: 8.5, color: 'rgba(184,220,210,0.72)' } as Partial<Plotly.Font>,
    } as Partial<Plotly.PlotData>

    // Dark background so the warm scale's near-white high end stays visible (white-on-white otherwise)
    // Flag the redraw so the plotly_restyle it emits (re-reporting our reapplied brushes) is ignored —
    // otherwise it would feed back into extractConstraints and fight the user's live selection.
    programmaticRedraw.current = true
    await Plotly.react(divRef.current, [trace],
      { paper_bgcolor: '#0a1422', plot_bgcolor: '#0a1422',
        margin: { l: 60, r: 130, t: 90, b: 20 }, height: pcHeight,
        font: { family: 'Epilogue, sans-serif', size: 10 } } as Partial<Plotly.Layout>,
      { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['toImage','lasso2d','select2d'] })
    // Re-enable user-brush capture once Plotly has emitted its post-react restyle events.
    window.setTimeout(() => { programmaticRedraw.current = false }, 80)

    // Attach the brush listener once per plot element. Plotly keeps gd event handlers across react()
    // calls, so re-adding per redraw would stack duplicates; we bind only when the element is new
    // (first draw, or after the div is remounted). A single cancellable timer debounces the drag, and
    // the handler reads fresh state via refs so a one-time binding stays correct.
    if (attachedEl.current !== divRef.current) {
      const gdEl = divRef.current as unknown as { on: (e: string, fn: () => void) => void }
      gdEl.on('plotly_restyle', () => {
        if (programmaticRedraw.current) return            // ignore our own reapplied brushes
        if (restyleTimer.current != null) window.clearTimeout(restyleTimer.current)
        restyleTimer.current = window.setTimeout(() => extractRef.current(), 60)
      })
      attachedEl.current = divRef.current
    }
  }

  function extractConstraints() {
    if (!divRef.current) return
    const gd  = divRef.current as unknown as {
      data?: { dimensions?: { label: string; constraintrange?: unknown }[] }[]
    }
    const dims = gd.data?.[0]?.dimensions
    if (!dims) return
    const c: Record<string, [number, number][]> = {}
    const snapBack: { di: number; value: unknown }[] = []   // categorical brushes snapped to whole cells

    dims.forEach((d, di) => {
      const cr = d.constraintrange
      if (!cr) return
      const rawRanges = (Array.isArray((cr as unknown[][])[0]) ? cr : [cr]) as [number, number][]
      const realCol = d.label
      const enc = data.catEncodings[realCol]

      if (enc) {
        // Categorical axis: each category i occupies the cell [i-0.5, i+0.5]. Select every category
        // whose cell the brush overlaps — so brushing anywhere over a category (including the half-cell
        // dead zone above the top / below the bottom one) reliably selects it. Without this, brushing
        // the rarest top category produces a band above its integer position → empty table.
        const n = enc.sorted.length
        const sel: number[] = []
        for (let i = 0; i < n; i++) {
          if (rawRanges.some(([lo, hi]) => i - 0.5 < hi && i + 0.5 > lo)) sel.push(i)
        }
        if (!sel.length) { snapBack.push({ di, value: null }); return }   // brush missed every cell → clear it
        // Merge selected indices into contiguous snapped ranges [first-0.5, last+0.5]
        const snapped: [number, number][] = []
        let start = sel[0], prev = sel[0]
        for (let k = 1; k < sel.length; k++) {
          if (sel[k] === prev + 1) prev = sel[k]
          else { snapped.push([start - 0.5, prev + 0.5]); start = prev = sel[k] }
        }
        snapped.push([start - 0.5, prev + 0.5])
        c[realCol] = snapped
        const snapVal = snapped.length === 1 ? snapped[0] : snapped
        const rawVal  = rawRanges.length === 1 ? rawRanges[0] : rawRanges
        if (JSON.stringify(rawVal) !== JSON.stringify(snapVal)) snapBack.push({ di, value: snapVal })
      } else if (rawRanges.length) {
        c[realCol] = rawRanges
      }
    })

    // Snap the categorical brushes visually so the highlighted lines match the (snapped) table filter.
    // Guarded by programmaticRedraw so the restyle's own plotly_restyle event is ignored.
    if (snapBack.length && plotRef.current && divRef.current) {
      programmaticRedraw.current = true
      const restyle = plotRef.current.restyle as (gd: Element, u: Record<string, unknown>) => unknown
      for (const s of snapBack) {
        try { restyle(divRef.current, { [`dimensions[${s.di}].constraintrange`]: [s.value] }) } catch { /* ignore */ }
      }
      window.setTimeout(() => { programmaticRedraw.current = false }, 80)
    }

    // Skip no-op updates (e.g. hover/redraw artefacts) so we don't re-render or recompute needlessly.
    if (sameConstraints(c, constraintsRef.current)) return
    setConstraints(c)
    onConstraintsChangeRef.current(c)
  }
  extractRef.current = extractConstraints

  // Redraw when active columns, color-by, sort mode, data, or 3D cluster selection changes
  useEffect(() => { drawPlot() }, [rows, colorByEff, sortMode, activeCols, activeCatCols, filter3D, removed]) // eslint-disable-line

  // Resize the plot to the new panel height during splitter drags — relayout (not redraw) so the
  // current axis brush is preserved.
  useEffect(() => {
    const Plotly = plotRef.current
    const el = divRef.current
    if (!Plotly || !el) return
    try {
      const p = Plotly.relayout(el, { height: pcHeight } as Partial<Plotly.Layout>)
      if (p && typeof (p as Promise<unknown>).catch === 'function') (p as Promise<unknown>).catch(() => {})
    } catch { /* plot not initialized yet — drawPlot will use the current height */ }
  }, [pcHeight])

  // ── Field picker helpers ───────────────────────────────────────────────────
  const toggleCol = (col: string) =>
    setSelectedCols(prev => {
      const next = new Set(prev); next.has(col) ? next.delete(col) : next.add(col); return next
    })

  const toggleCatCol = (col: string) =>
    setSelectedCatCols(prev => {
      const next = new Set(prev); next.has(col) ? next.delete(col) : next.add(col); return next
    })

  const selectAll  = () => setSelectedCols(new Set(data.numericCols))
  const selectNone = () => setSelectedCols(new Set())

  const visibleFieldCols = fieldSearch
    ? data.numericCols.filter(c => c.toLowerCase().includes(fieldSearch.toLowerCase()))
    : data.numericCols

  const visibleCatCols = fieldSearch
    ? data.categoricalCols.filter(c => c.toLowerCase().includes(fieldSearch.toLowerCase()))
    : data.categoricalCols

  // ── CSV export ─────────────────────────────────────────────────────────────
  const exportCSV = useCallback(() => {
    if (!filteredRows.length) return
    const esc  = (v: unknown) => {
      if (v == null) return ''
      const s = String(v)
      return (s.includes(',') || s.includes('"') || s.includes('\n')) ? `"${s.replace(/"/g, '""')}"` : s
    }
    // Bake committed partitions into the export as a cluster_id column.
    const withParts = !!partitions && partitions.map.size > 0
    const baseCols  = data.cols.map(c => c.name)
    const partCols  = withParts ? ['cluster_id'].filter(c => !baseCols.includes(c)) : []
    const cols      = [...baseCols, ...partCols]
    const cell = (r: CSVRow, c: string): unknown => {
      if (withParts && c === 'cluster_id' && partCols.includes(c)) {
        return partitions!.entryOf(r)?.cluster_id ?? ''
      }
      return r[c]
    }
    const csv  = [cols.map(esc).join(','), ...filteredRows.map(r => cols.map(c => esc(cell(r, c))).join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url  = URL.createObjectURL(blob)
    const a    = Object.assign(document.createElement('a'), {
      href: url,
      download: `selection_${filteredRows.length}rows_${data.fileName.replace(/\.csv$/i, '')}.csv`,
    })
    document.body.appendChild(a); a.click(); document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [filteredRows, data, partitions])

  const handleClusterSelect = useCallback((ids: Set<number> | null) => {
    setFilter3D(ids)
    if (ids !== null) {
      // Clear PC brush when a 3D cluster is selected so both don't conflict
      setConstraints({})
      onConstraintsChange({})
    }
  }, [onConstraintsChange])

  // Reset every parallel-coords axis brush + the 3D cluster selection.
  const resetFilters = useCallback(() => {
    const gd = divRef.current
    if (gd && plotRef.current) {
      const dims = (gd as unknown as { data?: { dimensions?: unknown[] }[] }).data?.[0]?.dimensions ?? []
      if (dims.length) {
        programmaticRedraw.current = true   // ignore the restyle's own events
        const restyle = plotRef.current.restyle as (g: Element, u: Record<string, unknown>) => unknown
        const update: Record<string, unknown> = {}
        dims.forEach((_, i) => { update[`dimensions[${i}].constraintrange`] = [null] })
        try { restyle(gd, update) } catch { /* ignore */ }
        window.setTimeout(() => { programmaticRedraw.current = false }, 80)
      }
    }
    if (restyleTimer.current != null) window.clearTimeout(restyleTimer.current)
    setConstraints({})
    onConstraintsChange({})
    setFilter3D(null)
  }, [onConstraintsChange])

  const hasActiveFilters = Object.keys(constraints).length > 0 || !!filter3D

  // ── Render ─────────────────────────────────────────────────────────────────
  const totalActive  = activeCols.length + activeCatCols.length
  const totalAxes    = data.numericCols.length + data.categoricalCols.length
  const unselected   = data.numericCols.length - selectedCols.size

  return (
    <div ref={rootRef} style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0,
                                overflowY: 'auto', overflowX: 'hidden' }}>

      {/* ── Action bar — only rendered when there's an active 3D selection ── */}
      {filter3D && (
      <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, padding: '7px 14px',
                    flexShrink: 0, borderBottom: '1px solid var(--c-border)', background: 'var(--c-raised)' }}>

        {/* 3D selection badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 10px',
                      borderRadius: 7, fontSize: 11,
                      background: 'rgba(41,204,165,0.12)',
                      border: '1px solid rgba(109,242,163,0.35)',
                      color: '#0a8f6e' }}>
          <span style={{ opacity: 0.7 }}>3D:</span>
          <span style={{ fontWeight: 600 }}>{filteredRows.length.toLocaleString()} rows</span>
          <button
            onClick={() => handleClusterSelect(null)}
            title="Clear 3D selection"
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0 0 0 2px',
                     color: 'rgba(10,143,110,0.7)', display: 'flex', alignItems: 'center',
                     lineHeight: 1 }}>
            <X size={11} />
          </button>
        </div>

      </div>
      )}

      {/* ── Field picker panel ── */}
      {showFields && (
        <div style={{ flexShrink: 0, borderBottom: '1px solid var(--c-border)',
                      background: 'var(--c-surface)', padding: '10px 14px' }}>

          {/* Picker header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--c-t2)' }}>
              Select axes &nbsp;
              <span style={{ fontWeight: 400, color: 'var(--c-t3)' }}>
                {totalActive} of {totalAxes} shown
                {unselected > 0 && ` · ${unselected} numeric hidden`}
              </span>
            </span>
            <button onClick={selectAll}
                    style={{ fontSize: 10, padding: '2px 8px', borderRadius: 5, cursor: 'pointer',
                             background: 'var(--c-accent-light)', color: 'var(--c-accent)',
                             border: '1px solid var(--c-accent)', fontWeight: 600 }}>
              All
            </button>
            <button onClick={selectNone}
                    style={{ fontSize: 10, padding: '2px 8px', borderRadius: 5, cursor: 'pointer',
                             background: 'var(--c-raised)', color: 'var(--c-t3)',
                             border: '1px solid var(--c-border)' }}>
              None
            </button>
            <input
              placeholder="Search fields…"
              value={fieldSearch}
              onChange={e => setFieldSearch(e.target.value)}
              style={{ fontSize: 11, padding: '2px 8px', borderRadius: 6, outline: 'none', width: 150,
                       background: 'var(--c-raised)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }}
            />
            {fieldSearch && (
              <button onClick={() => setFieldSearch('')} style={{ background: 'none', border: 'none',
                       cursor: 'pointer', color: 'var(--c-t3)', padding: 0 }}>
                <X size={12} />
              </button>
            )}
            {/* Reset all axis brushes + the 3D cluster selection — lives here in the Fields panel */}
            {hasActiveFilters && (
              <button onClick={resetFilters} title="Clear all parallel-coords filters and the 3D selection"
                      style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 5,
                               padding: '3px 9px', borderRadius: 6, cursor: 'pointer', fontSize: 11, fontWeight: 600,
                               background: 'var(--c-surface)', color: 'var(--c-t1)', border: '1px solid var(--c-border)' }}>
                <RotateCcw size={12} /> Reset filters
              </button>
            )}
            <button onClick={() => setShowFields(false)}
                    style={{ marginLeft: hasActiveFilters ? 0 : 'auto', background: 'none',
                     border: 'none', cursor: 'pointer', color: 'var(--c-t3)', padding: 2 }}>
              <X size={14} />
            </button>
          </div>

          {/* Numeric checkbox grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
                        gap: '3px 10px', maxHeight: 140, overflowY: 'auto' }}>
            {visibleFieldCols.map(col => {
              const checked = selectedCols.has(col)
              return (
                <label key={col}
                       style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer',
                                padding: '3px 6px', borderRadius: 5,
                                background: checked ? 'var(--c-hover)' : 'transparent' }}>
                  <input type="checkbox" checked={checked} onChange={() => toggleCol(col)}
                         style={{ accentColor: 'var(--c-accent)', width: 12, height: 12, cursor: 'pointer' }} />
                  <ColTooltip col={col}
                              style={{ fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis',
                                       whiteSpace: 'nowrap', color: checked ? 'var(--c-t1)' : 'var(--c-t3)' }}>
                    {col}
                  </ColTooltip>
                </label>
              )
            })}
            {visibleFieldCols.length === 0 && (
              <span style={{ fontSize: 11, color: 'var(--c-t3)', gridColumn: '1/-1' }}>No columns match</span>
            )}
          </div>

          {/* Categorical axes section */}
          {visibleCatCols.length > 0 && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, margin: '8px 0 4px',
                            borderTop: '1px solid var(--c-border)', paddingTop: 8 }}>
                <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase',
                               letterSpacing: '.5px', color: 'var(--c-t3)' }}>Categorical</span>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '3px 8px' }}>
                {visibleCatCols.map(col => {
                  const checked = selectedCatCols.has(col)
                  const enc     = data.catEncodings[col]
                  return (
                    <label key={col}
                           style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer',
                                    padding: '3px 8px', borderRadius: 12,
                                    background: checked ? 'var(--c-accent)' : 'var(--c-raised)',
                                    border: `1px solid ${checked ? 'var(--c-accent)' : 'var(--c-border)'}` }}>
                      <input type="checkbox" checked={checked} onChange={() => toggleCatCol(col)}
                             style={{ accentColor: '#fff', width: 11, height: 11, cursor: 'pointer' }} />
                      <ColTooltip col={col}
                                  style={{ fontSize: 11, whiteSpace: 'nowrap',
                                           color: checked ? '#fff' : 'var(--c-t2)' }}>
                        {col}
                      </ColTooltip>
                      {enc && (
                        <span style={{ fontSize: 10, opacity: 0.7,
                                       color: checked ? 'rgba(255,255,255,0.7)' : 'var(--c-t3)' }}>
                          {enc.sorted.length} values
                        </span>
                      )}
                    </label>
                  )
                })}
              </div>
            </>
          )}
        </div>
      )}

      {/* ── PC Chart ── */}
      {totalActive === 0 ? (
        <div style={{ height: pcHeight, flexShrink: 0, display: 'flex', alignItems: 'center',
                      justifyContent: 'center', background: '#f8fafc' }}>
          <p style={{ fontSize: 12, color: 'var(--c-t3)' }}>Select at least one field above to draw the chart</p>
        </div>
      ) : (
        <div ref={divRef} style={{ height: pcHeight, flexShrink: 0, background: '#0a1422' }} />
      )}

      {/* ── Splitter — drag to resize the top (parallel coords) vs bottom (3D + table) rows ── */}
      <div
        onMouseDown={startResize}
        title="Drag to resize"
        style={{ height: 8, flexShrink: 0, cursor: 'row-resize', position: 'relative',
                 background: 'var(--c-border)',
                 display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        onMouseEnter={e => (e.currentTarget.style.background = 'var(--c-accent)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'var(--c-border)')}
      >
        <div style={{ width: 38, height: 3, borderRadius: 2, background: 'rgba(255,255,255,0.55)',
                      pointerEvents: 'none' }} />
      </div>

      {/* ── Bottom split: 3D graph (left) + results table (right) ──
          minHeight keeps both panes usable on short screens; the explorer root scrolls if needed. */}
      <div style={{ flex: 1, minHeight: 280, display: 'flex', overflow: 'hidden' }}>

        {/* Left — 3D graph */}
        <div style={{ width: '50%', flexShrink: 0, borderRight: '2px solid var(--c-border)',
                      display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Graph3D
            data={data}
            rows={rows}
            embeddingsAvailable={embeddingsAvailable}
            datasetSource={datasetSource}
            selectedRows={filteredRows}
            colorBy={colorByEff}
            onClusterSelect={handleClusterSelect}
            onPointClick={row => onPlay(row, filteredRows)}
            partitions={partitions}
            onCommitClusterColumn={onCommitClusterColumn}
          />
        </div>

        {/* Right — results */}
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <ResultsSummary data={data} rows={filteredRows} total={rows.length}
                          activeConstraints={Object.keys(constraints).filter(k => selectedCols.has(k) || selectedCatCols.has(k)).length} />
          <ResultsTable data={data} rows={filteredRows} partitions={partitions} onPlay={row => onPlay(row, filteredRows)} />
        </div>

      </div>
    </div>
  )
}

// ── Summary bar ────────────────────────────────────────────────────────────────
function ResultsSummary({ data, rows, total, activeConstraints }:
  { data: CSVData; rows: CSVRow[]; total: number; activeConstraints: number }) {
  const pct = total ? ((rows.length / total) * 100).toFixed(1) : '0'
  return (
    <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 6, padding: '7px 12px',
                  flexShrink: 0, background: 'var(--c-raised)', borderBottom: '1px solid var(--c-border)' }}>
      <MiniCard label="rows"  value={rows.length.toLocaleString()} accent />
      <MiniCard label="total" value={total.toLocaleString()} />
      <MiniCard label="match" value={pct + '%'} />
      {activeConstraints > 0 && <MiniCard label="axes" value={`${activeConstraints} active`} />}
    </div>
  )
}

function MiniCard({ label, value, accent, mono }:
  { label: string; value: string; accent?: boolean; mono?: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center',
                  padding: '2px 8px', borderRadius: 7,
                  background: accent ? 'var(--c-accent)' : 'var(--c-surface)',
                  border: '1px solid var(--c-border)' }}>
      <span style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: '.5px',
                     color: accent ? 'rgba(255,255,255,0.7)' : 'var(--c-t3)' }}>{label}</span>
      <span style={{ fontSize: 12, fontWeight: 600, lineHeight: 1.2,
                     fontFamily: mono ? 'JetBrains Mono, monospace' : undefined,
                     color: accent ? '#fff' : 'var(--c-t1)' }}>{value}</span>
    </div>
  )
}

// ── Results mini-table ─────────────────────────────────────────────────────────
const MAX_ROWS = 300

function ResultsTable({ data, rows, partitions, onPlay }:
  { data: CSVData; rows: CSVRow[]; partitions?: Partitions; onPlay: (r: CSVRow) => void }) {

  const [sortCol, setSortCol] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const showPartition = !!partitions && partitions.map.size > 0

  const displayCols = useMemo(() => {
    const cols: string[] = []
    if (data.stemCol)  cols.push(data.stemCol)
    if (data.batchCol && data.batchCol !== data.stemCol) cols.push(data.batchCol)
    data.categoricalCols.forEach(c => { if (!cols.includes(c)) cols.push(c) })
    data.numericCols.slice(0, 8).forEach(c => { if (!cols.includes(c)) cols.push(c) })
    return cols
  }, [data])

  const sorted = useMemo(() => {
    if (!sortCol) return rows
    return [...rows].sort((a, b) => {
      const va = a[sortCol], vb = b[sortCol]
      if (va == null && vb == null) return 0
      if (va == null) return 1; if (vb == null) return -1
      if (typeof va === 'number' && typeof vb === 'number')
        return sortDir === 'asc' ? va - vb : vb - va
      return sortDir === 'asc' ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va))
    })
  }, [rows, sortCol, sortDir])

  const shown = sorted.slice(0, MAX_ROWS)

  const toggleSort = (col: string) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(col); setSortDir('desc') }
  }

  if (!rows.length) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: 'var(--c-t3)', fontSize: 12 }}>Brush axes or Ctrl+Click a cluster in 3D to filter rows</p>
      </div>
    )
  }

  return (
    <div style={{ flex: 1, overflowY: 'auto', overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: 'max-content', minWidth: '100%' }}>
        <thead>
          <tr style={{ position: 'sticky', top: 0, zIndex: 5 }}>
            {data.imagePath && <th style={thS} />}
            <th style={{ ...thS, fontSize: 10, color: 'rgba(255,255,255,0.35)', width: 40 }}>#</th>
            {showPartition && <th style={{ ...thS, width: 90 }}>partition</th>}
            {displayCols.map(col => (
              <th key={col} style={{ ...thS, cursor: 'pointer', background: sortCol === col ? TABLE_HEADER_ACTIVE : TABLE_HEADER }}
                  onClick={() => toggleSort(col)}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                  <span style={{ maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                        title={col}>{col}</span>
                  <span style={{ opacity: sortCol === col ? 1 : 0.3, flexShrink: 0 }}>
                    {sortCol !== col ? <ChevronsUpDown size={10} /> :
                     sortDir === 'asc' ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
                  </span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((row, ri) => (
            <tr key={row._idx as number}
                style={{ borderBottom: '1px solid var(--c-border)' }}
                onMouseEnter={e => (e.currentTarget.style.background = 'var(--c-hover)')}
                onMouseLeave={e => (e.currentTarget.style.background = '')}>
              {data.imagePath && (
                <td style={tdS}>
                  <button className="play-btn" onClick={() => onPlay(row)}>
                    <Play size={10} fill="currentColor" />
                  </button>
                </td>
              )}
              <td style={{ ...tdS, fontSize: 10, color: 'var(--c-t3)', textAlign: 'right', width: 40 }}>
                {ri + 1}
              </td>
              {showPartition && (
                <td style={tdS}><PartitionBadge entry={partitions!.entryOf(row)} /></td>
              )}
              {displayCols.map(col => (
                <td key={col} style={tdS}>
                  <CellValue value={row[col]} col={col} data={data} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > MAX_ROWS && (
        <p style={{ textAlign: 'center', padding: '6px 12px', fontSize: 11, color: 'var(--c-t3)',
                    borderTop: '1px solid var(--c-border)' }}>
          Showing {MAX_ROWS} of {rows.length.toLocaleString()} rows
        </p>
      )}
    </div>
  )
}

// Cluster id chip for the results table
function PartitionBadge({ entry }: { entry?: { cluster_id: number | null } }) {
  if (entry?.cluster_id != null) {
    return (
      <span style={{ padding: '1px 7px', borderRadius: 10, fontSize: 10, fontWeight: 600,
                     background: 'rgba(41,204,165,0.14)', color: 'var(--c-accent)' }}>
        Cluster {entry.cluster_id}
      </span>
    )
  }
  return <span style={{ color: 'var(--c-border-2)' }}>—</span>
}

function CellValue({ value, col, data }: { value: unknown; col: string; data: CSVData }) {
  if (value == null || value === '') return <span style={{ color: 'var(--c-border-2)' }}>—</span>
  if (typeof value === 'number') {
    return <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>
      {Number.isInteger(value) ? value.toLocaleString() : value.toPrecision(4).replace(/\.?0+$/, '')}
    </span>
  }
  const str   = String(value)
  const isCat = data.categoricalCols.includes(col)
  if (isCat) {
    return (
      <span style={{ padding: '1px 7px', borderRadius: 10, fontSize: 10, fontWeight: 600,
                     background: 'var(--c-accent-light)', color: 'var(--c-accent)' }}>
        {str}
      </span>
    )
  }
  const isPath = str.startsWith('/')
  return (
    <span style={{ display: 'block', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis',
                   whiteSpace: 'nowrap', fontSize: isPath ? 10 : 12,
                   fontFamily: isPath ? 'monospace' : undefined,
                   color: isPath ? 'var(--c-t3)' : 'var(--c-t1)' }} title={str}>
      {isPath ? str.split('/').pop() : str}
    </span>
  )
}

const thS: CSSProperties = {
  padding: '7px 10px', whiteSpace: 'nowrap', fontSize: 11, fontWeight: 500,
  textAlign: 'left', color: 'rgba(255,255,255,0.72)', background: TABLE_HEADER,
  borderBottom: '1px solid rgba(255,255,255,0.06)', letterSpacing: '.3px',
}
const tdS: CSSProperties = {
  padding: '4px 10px', verticalAlign: 'middle', whiteSpace: 'nowrap', fontSize: 12,
}

function fmtN(n: number) {
  if (!isFinite(n)) return '—'
  if (Math.abs(n) >= 10000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(2)
  return n.toPrecision(3).replace(/\.?0+$/, '')
}
