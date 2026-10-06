import { useCallback, useEffect, useRef, useState } from 'react'
import Papa from 'papaparse'
import { api } from '../api'
import type { CSVData } from '../types'
import type { RunSummary } from '../api'
import { buildFilter } from '../lib/experiments'
import { mergeRunColumns, removeRunColumns } from '../lib/mergeRunColumns'

const RUN_LIST_POLL_MS = 10_000

export interface Overlay { runId: string; epoch: number | 'latest' }
export interface Comparison { a: string; b: string }
export interface MatchInfo { matched: number; total: number }

/**
 * Merge a training run's per-sample columns into the loaded dataset.
 *
 * `data` is the dataset currently shown; `replaceData` swaps in the merged one. Every run linked
 * to `datasetSource` is listed (newest first in the picker); only those with a split or per-sample files can be picked.
 */
export function useRunOverlay(data: CSVData | null, datasetSource: string | null, replaceData: (d: CSVData) => void) {
  const [overlay, setOverlay] = useState<Overlay | null>(null)
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [match, setMatch] = useState<MatchInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [scoredEpochs, setScoredEpochs] = useState<number[]>([])
  const [compare, setCompare] = useState<Comparison | null>(null)
  const latestRequest = useRef(0)

  const refreshRuns = useCallback(async () => {
    if (!datasetSource) { setRuns([]); return }
    try {
      const page = await api.searchRuns({ experimentIds: [], filter: buildFilter({ dataset: datasetSource }), maxResults: 100 })
      setRuns(page.runs)
    } catch { setRuns([]) }
  }, [datasetSource])

  useEffect(() => { void refreshRuns() }, [refreshRuns])
  // Keep the picker current while something is queued or training; idle (and hidden) tabs do not poll.
  const live = runs.some(r => r.status === 'SCHEDULED' || r.status === 'RUNNING')
  useEffect(() => {
    if (!live) return
    const tick = () => { if (!document.hidden) void refreshRuns() }
    const id = setInterval(tick, RUN_LIST_POLL_MS)
    document.addEventListener('visibilitychange', tick)
    return () => { clearInterval(id); document.removeEventListener('visibilitychange', tick) }
  }, [live, refreshRuns])
  // A different dataset starts without an overlay.
  useEffect(() => { setOverlay(null); setCompare(null); setMatch(null); setError(null) }, [datasetSource, data?.fileName])

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
      setCompare(null)
      setScoredEpochs(detail.scored_epochs)
      setMatch({ matched: result.matched, total: result.matched + result.unmatched })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (request === latestRequest.current) setLoading(false)
    }
  }, [data, replaceData])

  /** Overlay the per-image change between two runs (`cmp_*` columns), replacing any run overlay. */
  const selectCompare = useCallback(async (a: string, b: string) => {
    if (!data) return
    const request = ++latestRequest.current
    setLoading(true)
    setError(null)
    try {
      const text = await api.getCompareSamples(a, b)
      if (request !== latestRequest.current) return
      const parsed = Papa.parse<Record<string, unknown>>(text, { header: true, dynamicTyping: true, skipEmptyLines: true })
      const result = mergeRunColumns(data, parsed.data, data.imagePath, 'cmp_')
      replaceData(result.data)
      setOverlay(null)
      setScoredEpochs([])
      setCompare({ a, b })
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
    setCompare(null)
    setLoading(false)
  }, [data, replaceData])

  return { overlay, compare, runs, match, loading, error, scoredEpochs, select, selectCompare, clear, refreshRuns }
}
