// Diverging scale for signed change columns (e.g. cmp_hardness_change): blue = went down (got easier),
// near-white at 0, orange-red = went up (got harder). Colour-blind safe, and the midpoint stays
// visible on the dark viz canvas.
export const DIVERGING_STOPS: { t: number; r: number; g: number; b: number }[] = [
  { t: 0,    r: 44,  g: 108, b: 196 },
  { t: 0.25, r: 126, g: 172, b: 226 },
  { t: 0.5,  r: 232, g: 232, b: 232 },
  { t: 0.75, r: 240, g: 160, b: 92  },
  { t: 1,    r: 214, g: 74,  b: 28  },
]

export const DIVERGING_SCALE: [number, string][] = DIVERGING_STOPS.map(s => [s.t, `rgb(${s.r},${s.g},${s.b})`])

/** Signed "change" columns are drawn on the diverging scale, centred on zero. */
export function isDivergingColumn(col: string): boolean {
  return col.endsWith('_change') || col.endsWith('_delta')
}

/** Symmetric colour range ±m covering the 98th percentile of |value|, so 0 sits at the centre. */
export function divergingRange(values: number[]): [number, number] {
  const abs = values.filter(Number.isFinite).map(Math.abs).sort((a, b) => a - b)
  if (!abs.length) return [-1, 1]
  const m = abs[Math.min(abs.length - 1, Math.round(0.98 * (abs.length - 1)))] || abs[abs.length - 1] || 1
  return [-m, m]
}

/** Colour for t in 0..1 as RGB floats (0–1), for the WebGL point buffer. */
export function interpolateDiverging(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  for (let i = 0; i < DIVERGING_STOPS.length - 1; i++) {
    const a = DIVERGING_STOPS[i], b = DIVERGING_STOPS[i + 1]
    if (t <= b.t) {
      const s = (t - a.t) / (b.t - a.t)
      return [(a.r + s * (b.r - a.r)) / 255, (a.g + s * (b.g - a.g)) / 255, (a.b + s * (b.b - a.b)) / 255]
    }
  }
  const l = DIVERGING_STOPS[DIVERGING_STOPS.length - 1]
  return [l.r / 255, l.g / 255, l.b / 255]
}
