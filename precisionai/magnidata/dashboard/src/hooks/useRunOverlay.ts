import { useCallback, useEffect, useRef, useState } from 'react'
import Papa from 'papaparse'
import { api } from '../api'
import type { CSVData } from '../types'
import type { RunSummary } from '../api'
import { buildFilter } from '../lib/experiments'
import { mergeRunColumns, removeRunColumns } from '../lib/mergeRunColumns'

export interface Overlay { runId: string; epoch: number | 'latest' }
export interface MatchInfo { matched: number; total: number }

/**
 * Merge a training run's per-sample columns into the loaded dataset.
 *
 * `data` is the dataset currently shown; `replaceData` swaps in the merged one. The runs
 * offered are those with per-sample artifacts linked to `datasetSource`.
 */
export function useRunOverlay(data: CSVData | null, datasetSource: string | null, replaceData: (d: CSVData) => void) {
  const [overlay, setOverlay] = useState<Overlay | null>(null)
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [match, setMatch] = useState<MatchInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [scoredEpochs, setScoredEpochs] = useState<number[]>([])
  const latestRequest = useRef(0)

  const refreshRuns = useCallback(async () => {
    if (!datasetSource) { setRuns([]); return }
    try {
      const page = await api.searchRuns({ experimentIds: [], filter: buildFilter({ dataset: datasetSource }), maxResults: 100 })
      setRuns(page.runs.filter(r => r.has_per_sample))
    } catch { setRuns([]) }
  }, [datasetSource])

  useEffect(() => { void refreshRuns() }, [refreshRuns])
  // A different dataset starts without an overlay.
  useEffect(() => { setOverlay(null); setMatch(null); setError(null) }, [datasetSource, data?.fileName])

  const select = useCallback(async (runId: string, epoch: number | 'latest' = 'latest') => {
    if (!data) return
    const request = ++latestRequest.current
    setLoading(true)
    setError(null)
    try {
      const [text, detail] = await Promise.all([api.getRunColumns(runId, epoch), api.getRun(runId)])
      // A newer selection (the scrubber moved on) supersedes this one; keep showing what is on screen.
      if (request !== latestRequest.current) return
      const parsed = Papa.parse<Record<string, unknown>>(text, { header: true, dynamicTyping: true, skipEmptyLines: true })
      const result = mergeRunColumns(data, parsed.data, data.imagePath)
      replaceData(result.data)
      setOverlay({ runId, epoch })
      setScoredEpochs(detail.scored_epochs)
      setMatch({ matched: result.matched, total: result.matched + result.unmatched })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (request === latestRequest.current) setLoading(false)
    }
  }, [data, replaceData])

  const clear = useCallback(() => {
    latestRequest.current++
    if (data) replaceData(removeRunColumns(data))
    setOverlay(null)
    setMatch(null)
    setError(null)
    setScoredEpochs([])
    setLoading(false)
  }, [data, replaceData])

  return { overlay, runs, match, loading, error, scoredEpochs, select, clear, refreshRuns }
}
