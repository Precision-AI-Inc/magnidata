/** The scored epoch closest to `value` (ties go to the earlier one); null when nothing is scored. */
export function nearestScoredEpoch(scored: number[], value: number): number | null {
  if (!scored.length) return null
  return scored.reduce((best, e) => (Math.abs(e - value) < Math.abs(best - value) ? e : best), scored[0])
}

/** The latest epoch with a prediction at or before `selected`; null when none exists. */
export function nearestEarlierPrediction(predictionEpochs: number[], selected: number): number | null {
  const earlier = predictionEpochs.filter(e => e <= selected)
  return earlier.length ? Math.max(...earlier) : null
}

/** The epoch after `current` in `scored`, wrapping to the first (for the play button). */
export function nextEpoch(scored: number[], current: number): number | null {
  if (!scored.length) return null
  return scored.find(e => e > current) ?? scored[0]
}

export function resolveEpoch(scored: number[], epoch: number | 'latest'): number | null {
  return epoch === 'latest' ? (scored[scored.length - 1] ?? null) : epoch
}
