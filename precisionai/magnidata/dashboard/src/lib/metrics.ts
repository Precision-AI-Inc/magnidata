export interface MetricPoint { step: number; value: number; timestamp: number }

const GROUP_ORDER = ['train_loss/', 'val_loss/', 'val_metric/', 'Learning_rate/']

export function groupName(key: string): string {
  return GROUP_ORDER.find(p => key.startsWith(p)) ?? 'other'
}

/** Group metric keys by their prefix (`train_loss/`, `val_loss/`, `val_metric/`, `Learning_rate/`, other); empty groups omitted. */
export function groupMetricKeys(keys: string[]): { group: string; keys: string[] }[] {
  const groups = new Map<string, string[]>()
  for (const key of [...keys].sort()) {
    const g = groupName(key)
    groups.set(g, [...(groups.get(g) ?? []), key])
  }
  return [...GROUP_ORDER, 'other'].filter(g => groups.has(g)).map(g => ({ group: g, keys: groups.get(g)! }))
}

export interface TrainValPair { term: string; train?: string; val?: string }

/** Pair `train_loss/<t>` with `val_loss/<t>`; terms present on one side only keep the other undefined. */
export function pairTrainVal(keys: string[]): TrainValPair[] {
  const pairs = new Map<string, TrainValPair>()
  for (const key of keys) {
    const train = key.startsWith('train_loss/')
    const val = key.startsWith('val_loss/')
    if (!train && !val) continue
    const term = key.slice(key.indexOf('/') + 1)
    const pair = pairs.get(term) ?? { term }
    if (train) pair.train = key
    else pair.val = key
    pairs.set(term, pair)
  }
  // The weighted total ("loss") first, then the other terms alphabetically.
  return [...pairs.values()].sort((a, b) => (a.term === 'loss' ? -1 : b.term === 'loss' ? 1 : a.term.localeCompare(b.term)))
}

/** Exponential moving average; alpha 0 returns the series unchanged, values near 1 smooth heavily. */
export function ema(series: number[], alpha: number): number[] {
  if (alpha <= 0) return [...series]
  const out: number[] = []
  let prev = 0
  series.forEach((v, i) => {
    prev = i === 0 ? v : alpha * prev + (1 - alpha) * v
    out.push(prev)
  })
  return out
}

/** Seconds since the first point, for the "relative time" x axis. */
export function toRelativeTime(points: MetricPoint[]): number[] {
  if (!points.length) return []
  const t0 = points[0].timestamp
  return points.map(p => (p.timestamp - t0) / 1000)
}
