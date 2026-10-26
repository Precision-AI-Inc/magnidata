/**
 * Evenly sample an ordered row set, retaining both endpoints. Parallel-coordinate
 * brushing derives numeric constraints from the sample, but those constraints are
 * subsequently evaluated against the complete dataset.
 */
export function sampleParallelRows<T>(rows: T[], limit: number): T[] {
  if (limit <= 0 || rows.length <= limit) return rows
  if (limit === 1) return [rows[0]]
  const step = (rows.length - 1) / (limit - 1)
  return Array.from({ length: limit }, (_, index) => rows[Math.round(index * step)])
}
