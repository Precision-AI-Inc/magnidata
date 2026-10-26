export interface ExperimentCount {
  total: number
  active: number
}

export interface ExperimentBadge {
  /** `active` renders yellow, `idle` gray. */
  state: 'active' | 'idle'
  label: string
}

const plural = (count: number, word: string) => `${count} ${word}${count === 1 ? '' : 's'}`

/** Badge shown on a dataset card: yellow while any run is queued or training, gray otherwise. */
export function experimentBadge(count: ExperimentCount | undefined): ExperimentBadge {
  if (!count || count.total === 0) return { state: 'idle', label: 'No experiments' }
  if (count.active > 0) return { state: 'active', label: `${count.active} running · ${plural(count.total, 'experiment')} total` }
  return { state: 'idle', label: `${plural(count.total, 'experiment')}, none running` }
}

export function anyActive(counts: Record<string, ExperimentCount>): boolean {
  return Object.values(counts).some(count => count.active > 0)
}
