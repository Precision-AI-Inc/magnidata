import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../api'
import type { Experiment, MlflowStatus, RunStatus, RunSummary } from '../api'
import { buildFilter, buildOrderBy, defaultSelection, DEFAULT_METRICS, DEFAULT_PARAMS } from '../lib/experiments'
import type { ColumnKind } from '../lib/experiments'

const COLUMNS_KEY = 'magnidata.experiments.columns'

export interface SortState { kind: ColumnKind; key: string; dir: 'asc' | 'desc' }
export interface ChosenColumns { metrics: string[]; params: string[] }

function loadColumns(): ChosenColumns {
  try {
    const raw = localStorage.getItem(COLUMNS_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed.metrics) && Array.isArray(parsed.params)) return parsed
    }
  } catch { /* storage unavailable or corrupt: fall back to defaults */ }
  return { metrics: DEFAULT_METRICS, params: DEFAULT_PARAMS }
}

/** Experiments, the run table's data and its filter/sort/column state. */
export function useExperiments(currentDataset: string | null) {
  const [mlflow, setMlflow] = useState<MlflowStatus | null>(null)
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [experimentQuery, setExperimentQuery] = useState('')
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [nextToken, setNextToken] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [status, setStatus] = useState<RunStatus | null>(null)
  const [onlyDataset, setOnlyDataset] = useState(false)
  const [text, setText] = useState('')
  const [sort, setSort] = useState<SortState>({ kind: 'attribute', key: 'start', dir: 'desc' })
  const [columns, setColumnsState] = useState<ChosenColumns>(loadColumns)
  const initialised = useRef(false)

  const setColumns = useCallback((next: ChosenColumns) => {
    setColumnsState(next)
    try { localStorage.setItem(COLUMNS_KEY, JSON.stringify(next)) } catch { /* ignore */ }
  }, [])

  const loadExperiments = useCallback(async () => {
    try {
      const [st, list] = await Promise.all([api.getMlflowStatus(), api.listExperiments(experimentQuery)])
      setMlflow(st)
      setExperiments(list.experiments)
      if (!initialised.current && !experimentQuery) {
        initialised.current = true
        setSelected(defaultSelection(list.experiments))
      }
      setError(null)
    } catch (e) {
      if (e instanceof ApiError && e.status === 503) {
        setMlflow(prev => ({ ...(prev ?? { tracking_uri: null, ui_url: null, version: null }), reachable: false, error: 'MLflow is unreachable' }))
      } else {
        setError(e instanceof Error ? e.message : String(e))
      }
    }
  }, [experimentQuery])

  const filter = buildFilter({ status, dataset: onlyDataset ? currentDataset : null, text })
  const orderBy = buildOrderBy(sort.kind, sort.key, sort.dir)

  const loadRuns = useCallback(async (append: boolean, token: string | null) => {
    if (!selected.length) { setRuns([]); setNextToken(null); return }
    setLoading(true)
    try {
      const page = await api.searchRuns({ experimentIds: selected, filter, orderBy: orderBy ?? undefined, pageToken: token })
      setRuns(prev => (append ? [...prev, ...page.runs] : page.runs))
      setNextToken(page.next_page_token)
      setError(null)
    } catch (e) {
      setRuns([])
      setNextToken(null)
      setError(e instanceof ApiError && e.status === 503 ? null : e instanceof Error ? e.message : String(e))
      if (e instanceof ApiError && e.status === 503) setMlflow(prev => prev && { ...prev, reachable: false })
    } finally {
      setLoading(false)
    }
  }, [selected, filter, orderBy])

  useEffect(() => { void loadExperiments() }, [loadExperiments])
  useEffect(() => { void loadRuns(false, null) }, [loadRuns])

  const refresh = useCallback(() => { void loadExperiments(); void loadRuns(false, null) }, [loadExperiments, loadRuns])
  const loadMore = useCallback(() => { if (nextToken) void loadRuns(true, nextToken) }, [loadRuns, nextToken])
  const toggleExperiment = useCallback((id: string) => {
    setSelected(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]))
  }, [])

  return {
    mlflow, experiments, selected, toggleExperiment, experimentQuery, setExperimentQuery,
    runs, hasMore: nextToken != null, loadMore, loading, error, refresh,
    status, setStatus, onlyDataset, setOnlyDataset, text, setText, sort, setSort, columns, setColumns,
  }
}
