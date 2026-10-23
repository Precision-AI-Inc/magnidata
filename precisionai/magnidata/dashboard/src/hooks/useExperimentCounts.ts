import { useEffect, useState } from 'react'
import { api } from '../api'
import type { ExperimentCounts } from '../api'
import { anyActive } from '../lib/experimentBadge'

const REFRESH_MS = 10_000

/** Per-dataset run counts, re-polled every 10 s while any run is queued or training. */
export function useExperimentCounts(): ExperimentCounts {
  const [counts, setCounts] = useState<ExperimentCounts>({})
  const live = anyActive(counts)

  useEffect(() => {
    let cancelled = false
    const load = () => {
      api.datasetExperimentCounts()
        .then(next => { if (!cancelled) setCounts(next) })
        .catch(() => { /* badges just stay gray while the API is unreachable */ })
    }
    load()
    if (!live) return () => { cancelled = true }
    const timer = window.setInterval(load, REFRESH_MS)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [live])

  return counts
}
