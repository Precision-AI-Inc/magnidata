import { useEffect, useState } from 'react'
import { api, errorText } from '../api'
import type { DatasetSummary, RunSummary } from '../api'
import { buildFilter } from '../lib/experiments'

export interface DatasetDetail {
  summary: DatasetSummary | null
  summaryError: string | null
  runs: RunSummary[]
  runsLoading: boolean
}

const EMPTY: DatasetDetail = { summary: null, summaryError: null, runs: [], runsLoading: true }

/** Statistics and recent training runs for one dataset; resets whenever the dataset changes. */
export function useDatasetDetail(source: string, enabled = true): DatasetDetail {
  const [state, setState] = useState<{ source: string; detail: DatasetDetail }>({ source, detail: EMPTY })

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    const patch = (update: Partial<DatasetDetail>) => {
      if (!cancelled) setState(prev => ({ source, detail: { ...(prev.source === source ? prev.detail : EMPTY), ...update } }))
    }
    patch({ ...EMPTY })

    api.datasetSummary(source)
      .then(summary => patch({ summary }))
      .catch(error => patch({ summaryError: errorText(error) }))

    api.listExperiments()
      .then(async ({ experiments }) => {
        if (!experiments.length) return patch({ runs: [], runsLoading: false })
        const page = await api.searchRuns({
          experimentIds: experiments.map(e => e.experiment_id),
          filter: buildFilter({ dataset: source }),
          orderBy: 'attributes.start_time DESC',
          maxResults: 8,
        })
        patch({ runs: page.runs, runsLoading: false })
      })
      .catch(() => patch({ runs: [], runsLoading: false }))

    return () => { cancelled = true }
  }, [source, enabled])

  return state.source === source ? state.detail : EMPTY
}
