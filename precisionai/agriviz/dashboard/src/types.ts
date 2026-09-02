// ── CSV data ───────────────────────────────────────────────────────────────────
export interface CSVRow {
  _idx: number
  [key: string]: unknown
}

export type ColType = 'numeric' | 'categorical' | 'string'

export interface ColMeta {
  name: string
  type: ColType
  // numeric
  min?: number
  max?: number
  mean?: number
  std?: number
  p25?: number
  p50?: number
  p75?: number
  // categorical / string
  unique?: number
  topValues?: [string, number][]  // [value, count]
  nullCount: number
}

export interface CatEncoding {
  sorted: string[]                  // unique values sorted by frequency desc
  encode: Record<string, number>    // value → integer index
}

export interface CSVData {
  rows: CSVRow[]
  cols: ColMeta[]
  numericCols: string[]
  categoricalCols: string[]
  stringCols: string[]
  /** Categorical encoding map — used to render categorical PC axes */
  catEncodings: Record<string, CatEncoding>
  /** Detected special columns */
  imagePath: string | null   // column whose values look like image paths
  maskPath:  string | null
  thumbnailPath: string | null   // pre-generated small preview per row (e.g. a CDN thumbnail), if present
  stemCol:   string | null   // short display name column
  batchCol:  string | null
  clusterCol: string | null  // predetermined cluster/partition column, if present
  fileName:  string
}

// ── Filters ────────────────────────────────────────────────────────────────────
export interface FilterState {
  ranges:     Record<string, [number, number]>   // col → [lo, hi]
  categories: Record<string, Set<string>>         // col → selected values
  search:     string
  pcConstraints: Record<string, [number, number][]> // from parallel-coords brush
}

// ── Notes ──────────────────────────────────────────────────────────────────────
export interface Note {
  id: number
  stem: string
  image_path: string
  note: string
  created_at: string
  updated_at: string
}

// ── Tabs ───────────────────────────────────────────────────────────────────────
export type TabId = 'overview' | 'table' | 'visual-explorer' | 'charts' | 'schema'
