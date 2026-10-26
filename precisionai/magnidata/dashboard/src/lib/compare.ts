export type Direction = 'min' | 'max'

export interface ParamRow {
  key: string
  /** One value per run, in run order; undefined where the run does not have the param. */
  values: (string | undefined)[]
  differs: boolean
}

/** Rows of params across runs. A param missing from some runs counts as differing. */
export function diffParams(runs: { params: Record<string, string> }[], onlyDiffering = true): ParamRow[] {
  const keys = [...new Set(runs.flatMap(r => Object.keys(r.params)))].sort()
  const rows = keys.map(key => {
    const values = runs.map(r => r.params[key])
    return { key, values, differs: new Set(values).size > 1 }
  })
  return onlyDiffering ? rows.filter(r => r.differs) : rows
}

/** The best value of a series (smallest for 'min', largest for 'max'); undefined when empty. */
export function bestValue(series: number[], direction: Direction): number | undefined {
  const finite = series.filter(Number.isFinite)
  if (!finite.length) return undefined
  return direction === 'min' ? Math.min(...finite) : Math.max(...finite)
}

const HIGHER_IS_BETTER = /(iou|acc|f1|precision|recall|score|dice_coef)/i

/** Losses and unknown metrics are better when lower; IoU/accuracy-like metrics when higher. */
export function defaultDirection(key: string): Direction {
  if (/loss|error/i.test(key)) return 'min'
  return HIGHER_IS_BETTER.test(key) ? 'max' : 'min'
}

// Categorical palette with enough hue spread for 10 lines on a light or dark background.
const RUN_COLORS = [
  '#0E7C90', '#E08A1E', '#B5479C', '#4C78C8', '#6DB33F',
  '#D95F5F', '#7E57C2', '#8A6D3B', '#29A8A0', '#5F6B7A',
]

/** Same ids in the same order always get the same colours, shared by every chart and legend. */
export function assignRunColors(runIds: string[]): Record<string, string> {
  return Object.fromEntries(runIds.map((id, i) => [id, RUN_COLORS[i % RUN_COLORS.length]]))
}

const HASH_PREFIX = '#compare='

export function compareHash(runIds: string[]): string {
  return `${HASH_PREFIX}${runIds.join(',')}`
}

/** Run ids encoded in a `#compare=a,b` hash; empty when it is not a comparison hash. */
export function parseCompareHash(hash: string): string[] {
  if (!hash.startsWith(HASH_PREFIX)) return []
  return hash.slice(HASH_PREFIX.length).split(',').filter(id => /^[0-9a-f]{32}$/.test(id)).slice(0, 10)
}
