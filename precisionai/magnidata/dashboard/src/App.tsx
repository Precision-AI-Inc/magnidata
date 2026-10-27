import { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { Activity, ChevronLeft, ChevronRight, CloudDownload, Columns, Database, Download, FlaskConical, LayoutGrid, Loader2, Plus, Save, Share2, SlidersHorizontal, Table2, Trash2, X, BarChart2 } from 'lucide-react'
import { rememberWorkspaceSelection } from './lib/workspaceSelection'
import { api, errorText, setMlflowOffline } from './api'
import type { DatasetMeta } from './api'
import { useCSV } from './hooks/useCSV'
import { useFilters } from './hooks/useFilters'
import { usePartitions } from './hooks/usePartitions'
import { useBuildJob } from './hooks/useBuildJob'
import { useRunOverlay } from './hooks/useRunOverlay'
import { autoOverlayTarget, pickerEntries } from './lib/runAxes'
import { splitSummary } from './lib/split'
import { useActiveRun } from './hooks/useActiveRun'
import { TrainDialog } from './components/experiments/TrainDialog'
import { paramsFromRun, progressLabel, trainDisabledReason } from './lib/train'
import { DEMO_DISABLED_REASON } from './lib/demo'
import { DemoContext } from './hooks/useDemoReason'
import type { TrainParams } from './lib/train'
import { buildRelabelCsv } from './lib/relabel'
import { RunOverlayPicker } from './components/RunOverlayPicker'
import { EpochScrubber } from './components/EpochScrubber'
import type { CSVRow, TabId } from './types'

import { Header }         from './components/Header'
import { Sidebar }        from './components/Sidebar'
import { ByodDialog }     from './components/ByodDialog'
import { GuidePage }      from './components/guide/GuidePage'
import { DataTable }      from './components/DataTable'
import { Overview }       from './components/Overview'
import { ParallelCoords } from './components/ParallelCoords'
import { Histograms }     from './components/Histograms'
import { ImageModal }     from './components/ImageModal'
import { Schema }         from './components/Dictionary'
import { ExperimentsTab } from './components/experiments/ExperimentsTab'
import { AppNavigation } from './components/AppNavigation'
import type { AppSection } from './components/AppNavigation'
import { WorkspaceView } from './components/WorkspaceView'
import {
  forgetDatasetSource, readLastDatasetSource, rememberedDataset, rememberDatasetSource, retry,
} from './lib/datasetSession'
import './components/WorkspaceDashboard.css'

type LandingView = 'landing' | 'help' | 'experiments'

const CLOUD_DATASET_SOURCE = 'data_user/cloud-dataset.csv'

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'visual-explorer', label: 'Visual Explorer', icon: <Share2 size={13} /> },
  { id: 'charts',          label: 'Charts',          icon: <BarChart2 size={13} /> },
  { id: 'schema',          label: 'Schema',          icon: <LayoutGrid size={13} /> },
  { id: 'table',           label: 'Table',           icon: <Table2 size={13} /> },
  { id: 'overview',        label: 'Overview',        icon: <Activity size={13} /> },
  { id: 'experiments',     label: 'Experiments',     icon: <FlaskConical size={13} /> },
]

export default function App() {
  const [tab, setTab]           = useState<TabId>('visual-explorer')
  // Image preview holds the clicked row + the exact set of rows to navigate (the current selection)
  const [preview, setPreview]   = useState<{ row: CSVRow; rows: CSVRow[] } | null>(null)
  const [hiddenCols, setHiddenCols] = useState<Set<string>>(new Set())
  const [datasetSource, setDatasetSource] = useState<string | null>(null)
  const [datasetMeta, setDatasetMeta] = useState<DatasetMeta[]>([])
  const [catalogReady, setCatalogReady] = useState(false)
  const [restoringDataset, setRestoringDataset] = useState(() => readLastDatasetSource() !== null)
  const [fetchingDataset, setFetchingDataset] = useState(false)
  const [datasetLoadError, setDatasetLoadError] = useState<string | null>(null)
  const [failedDatasetSource, setFailedDatasetSource] = useState<string | null>(null)
  const restoreAttempted = useRef(false)
  const [landingView, setLandingView] = useState<LandingView>('landing')
  const [showByod, setShowByod] = useState(false)
  // True on a demo deployment, which ships its datasets preloaded: adding datasets, training
  // and experiments stay visible but are disabled, with DEMO_DISABLED_REASON as the tooltip.
  const [demoMode, setDemoMode] = useState(false)
  const demoReason = demoMode ? DEMO_DISABLED_REASON : undefined
  const [experimentsDataset, setExperimentsDataset] = useState<string | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [pendingOverlay, setPendingOverlay] = useState<{ runId: string; compareWith?: string; source: string } | null>(null)
  // A reloaded `#compare=…` link goes straight to the experiments view.
  useEffect(() => {
    if (window.location.hash.startsWith('#compare=')) setLandingView('experiments')
    else if (window.location.hash.startsWith('#guide=')) setLandingView('help')
  }, [])

  // Staged image removals (by row _idx). They only shape the live views in memory; the original
  // dataset is never touched. They "apply" only when saved as a new dataset (or discarded otherwise).
  const [removed, setRemoved] = useState<Set<number>>(new Set())
  const [showRemovalSave, setShowRemovalSave] = useState(false)
  const [removalName, setRemovalName] = useState('')
  const [savingRemoval, setSavingRemoval] = useState(false)
  const [removalMsg, setRemovalMsg] = useState<string | null>(null)

  // Hand-picked images (a curation "basket"). Saving turns the picked set into a brand-new dataset.
  // Visual-explorer plot controls — lifted up so they render in the tab bar (see below).
  const [colorBy, setColorByState] = useState('')
  // Once the user picks a colour themselves, an overlay no longer switches it to `split`.
  const userPickedColor = useRef(false)
  const setColorBy = useCallback((c: string | ((prev: string) => string)) => setColorByState(c), [])
  const chooseColorBy = useCallback((c: string) => { userPickedColor.current = true; setColorByState(c) }, [])
  const [sortMode, setSortMode]     = useState<'default' | 'variance' | 'corr'>('default')
  const [showFields, setShowFields] = useState(false)
  const [explorerPanelOpen, setExplorerPanelOpen] = useState(true)
  const [exportingSelection, setExportingSelection] = useState(false)

  const [added, setAdded] = useState<Set<number>>(new Set())
  const [showAddSave, setShowAddSave] = useState(false)
  const [addName, setAddName] = useState('')
  const [savingAdd, setSavingAdd] = useState(false)
  const [addMsg, setAddMsg] = useState<string | null>(null)
  const [complementName, setComplementName] = useState('')
  const [complementTopK, setComplementTopK] = useState(100)
  const [savingComplement, setSavingComplement] = useState(false)
  const [complementMsg, setComplementMsg] = useState<string | null>(null)
  const [selectionName, setSelectionName] = useState('')
  const [savingSelection, setSavingSelection] = useState(false)
  const [selectionMsg, setSelectionMsg] = useState<string | null>(null)
  const [savedSelection, setSavedSelection] = useState<DatasetMeta | null>(null)

  // Embeddings now live entirely on the API — the browser only needs to know they exist.
  const embeddingsAvailable = useMemo(
    () => !!datasetSource && !!datasetMeta.find(d => d.source === datasetSource)?.has_embeddings,
    [datasetSource, datasetMeta],
  )

  const { data, loading, error, load, setClusterColumn, clear, replaceData } = useCSV()
  const {
    filters, filteredRows,
    setRange, setCategory, setSearch, setPCConstraints,
    reset, activeCount,
  } = useFilters(data)

  // Persistent cluster partitions (cluster_id) for the active dataset — shared by the
  // 3D view, the results table, and the image preview.
  const partitions = usePartitions(datasetSource, data)

  const visibleData = useMemo(() => {
    if (!data || !hiddenCols.size) return data
    const keep = (name: string) => !hiddenCols.has(name)
    return {
      ...data,
      cols:           data.cols.filter(c => keep(c.name)),
      numericCols:    data.numericCols.filter(keep),
      categoricalCols: data.categoricalCols.filter(keep),
      stringCols:     data.stringCols.filter(keep),
    }
  }, [data, hiddenCols])

  // Keep colorBy valid for the current dataset: default to complexity_score, else first column.
  useEffect(() => {
    if (!data) return
    const cats = visibleData?.categoricalCols ?? data.categoricalCols
    const nums = visibleData?.numericCols ?? data.numericCols
    setColorBy(prev => (prev && [...cats, ...nums].includes(prev))
      ? prev
      : (nums.includes('complexity_score') ? 'complexity_score' : (cats[0] ?? nums[0] ?? '')))
  }, [data, visibleData])

  // A new dataset starts with the default colouring; the first overlay that brings a split colours by it.
  useEffect(() => { userPickedColor.current = false }, [data?.fileName])
  const hasSplitColumn = !!data?.runColumns?.includes('split')
  useEffect(() => {
    if (hasSplitColumn && !userPickedColor.current) setColorByState('split')
  }, [hasSplitColumn])

  // Filtered rows with the staged removals excluded — what the tabs/preview show. The full row set
  // (data.rows) is kept for the explorer so the server-side embedding projection stays aligned.
  const liveFilteredRows = useMemo(
    () => (removed.size ? filteredRows.filter(r => !removed.has(r._idx as number)) : filteredRows),
    [filteredRows, removed],
  )

  const exportSelection = useCallback(async () => {
    if (!data || !liveFilteredRows.length) return
    const baseName = (data.fileName || 'dataset').replace(/\.csv$/i, '')
    const cols = Object.keys(liveFilteredRows[0]).filter(key => key !== '_idx')
    const escape = (value: unknown) => {
      const text = value == null ? '' : String(value)
      return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
    }
    const csv = [cols.join(','), ...liveFilteredRows.map(row => cols.map(col => escape(row[col])).join(','))].join('\n')
    const download = (content: string, filename: string, type: string) => {
      const anchor = document.createElement('a')
      anchor.href = URL.createObjectURL(new Blob([content], { type }))
      anchor.download = filename
      anchor.click()
      URL.revokeObjectURL(anchor.href)
    }
    download(csv, `${baseName}_selection.csv`, 'text/csv')
    if (!embeddingsAvailable || !datasetSource || !data.imagePath) return
    setExportingSelection(true)
    try {
      const { embeddings } = await api.getEmbeddings(datasetSource)
      const keys = new Set(liveFilteredRows.map(row => String(row[data.imagePath!] ?? '')).filter(Boolean))
      const selected: Record<string, number[]> = {}
      for (const key of keys) if (embeddings[key]) selected[key] = embeddings[key]
      download(JSON.stringify({ embeddings: selected }), `${baseName}_selection_embeddings.json`, 'application/json')
    } catch {
      // The CSV has already downloaded; embeddings are an optional companion export.
    } finally {
      setExportingSelection(false)
    }
  }, [data, datasetSource, embeddingsAvailable, liveFilteredRows])

  const toggleRemove = useCallback((row: CSVRow) => {
    setRemoved(prev => {
      const next = new Set(prev)
      const id = row._idx as number
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }, [])
  const discardRemovals = useCallback(() => { setRemoved(new Set()); setShowRemovalSave(false); setRemovalMsg(null) }, [])

  const toggleAdd = useCallback((row: CSVRow) => {
    setAdded(prev => {
      const next = new Set(prev)
      const id = row._idx as number
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }, [])
  const clearAdded = useCallback(() => { setAdded(new Set()); setShowAddSave(false); setAddMsg(null) }, [])

  // Download just the picked images as a CSV (all columns). Works for any dataset, uploads included.
  const exportAdded = useCallback(() => {
    if (!data || !added.size) return
    const picked = data.rows.filter(r => added.has(r._idx as number))
    if (!picked.length) return
    const cols = data.cols.map(c => c.name)
    const esc = (v: unknown) => {
      if (v == null) return ''
      const s = String(v)
      return (s.includes(',') || s.includes('"') || s.includes('\n')) ? `"${s.replace(/"/g, '""')}"` : s
    }
    const csv = [cols.map(esc).join(','), ...picked.map(r => cols.map(c => esc(r[c])).join(','))].join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }))
    const a = Object.assign(document.createElement('a'), {
      href: url, download: `picked_${picked.length}rows_${data.fileName.replace(/\.csv$/i, '')}.csv`,
    })
    document.body.appendChild(a); a.click(); document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [data, added])

  const toggleCol = (col: string) =>
    setHiddenCols(prev => {
      const next = new Set(prev)
      next.has(col) ? next.delete(col) : next.add(col)
      return next
    })

  const reloadDatasets = useCallback(async () => {
    try {
      const datasets = await retry(() => api.datasets())
      setDatasetMeta(datasets)
      setCatalogReady(true)
      setDatasetLoadError(null)
      return datasets
    } catch (caught) {
      setCatalogReady(false)
      setRestoringDataset(false)
      setDatasetLoadError(`Could not load the dataset catalogue. ${errorText(caught)}`)
      return null
    }
  }, [])

  useEffect(() => {
    let alive = true
    api.config().then(config => {
      if (!alive) return
      setDemoMode(config.demo_mode === true)
      setMlflowOffline(config.demo_mode === true)   // a demo has no MLflow: stop polling for runs and presets
    }).catch(() => {})
    return () => { alive = false }
  }, [])

  const deleteDataset = useCallback((id: number) => {
    const source = datasetMeta.find(dataset => dataset.id === id)?.source
    api.deleteDataset(id).then(() => {
      if (source) forgetDatasetSource(source)
      void reloadDatasets()
    }).catch(() => {})
  }, [datasetMeta, reloadDatasets])

  // Global dataset-build progress — lives here (not inside the BYOD dialog) so it keeps
  // polling and stays visible (as a Header badge) after the dialog that started it closes.
  const handleBuildDone = useCallback((_dataset: DatasetMeta) => {
    reloadDatasets()
    setLandingView('landing')      // the finished dataset shows up in the lineage graph
  }, [reloadDatasets])
  const buildJob = useBuildJob(handleBuildDone)

  // Load the dataset catalogue (for embeddings availability) on mount
  useEffect(() => {
    void reloadDatasets()
  }, [reloadDatasets])

  const handleLoad = (f: File, source?: string) => {
    load(f)
    if (source) rememberDatasetSource(source)
    setDatasetLoadError(null)
    setFailedDatasetSource(null)
    setTab('visual-explorer')
    setPreview(null)
    reset()
    setHiddenCols(new Set())
    setRemoved(new Set())          // pending removals don't carry across loads
    setShowRemovalSave(false); setRemovalName(''); setRemovalMsg(null)
    setAdded(new Set())            // the curation basket is per-dataset
    setShowAddSave(false); setAddName(''); setAddMsg(null)
    setComplementName(''); setComplementMsg(null)
    // Embeddings are computed server-side now — the browser never downloads the vectors.
    setDatasetSource(source ?? null)
  }

  // Unload any loaded dataset and return to the 3-card landing screen — reachable from
  // anywhere (the Header nav is always mounted), not just from within Landing itself.
  const goHome = () => {
    clear()
    setDatasetSource(null)
    setTab('visual-explorer')
    setLandingView('landing')
  }

  // Same, but goes straight to the dataset picker instead of the landing cards — unless
  // no dataset has been prepared yet, in which case BYOD opens first so the user has
  // something to pick; handleBuildDone() already routes back into the picker once a
  // build finishes, so the MagniData flow continues from there.
  const goToMagniData = () => {
    clear()
    setDatasetSource(null)
    setTab('visual-explorer')
    if (datasetMeta.length === 0 && !demoMode) {
      setShowByod(true)
      return
    }
    setLandingView('landing')
  }

  const goToExplorer = () => {
    if (data) {
      setTab('visual-explorer')
      return
    }
    // Nothing loaded: open the last dataset used, else the Cloud Dataset, else the first usable one.
    const usable = datasetMeta.filter(d => !d.is_lfs_pointer)
    const target = rememberedDataset(usable, readLastDatasetSource())
      ?? usable.find(d => d.source === CLOUD_DATASET_SOURCE)
      ?? usable[0]
    if (target) {
      setLandingView('landing')
      void loadDatasetBySource(target.source)
      return
    }
    goToMagniData()
  }

  const goToSchema = () => {
    clear()
    setDatasetSource(null)
    setTab('schema')
  }

  // Help is a full page (not a dialog) — reachable the same way as Home/MagniData.
  const goToHelp = () => {
    clear()
    setDatasetSource(null)
    setTab('visual-explorer')
    setLandingView('help')
  }

  // "Load dataset" from a run: fetch the CSV of a known dataset and open it like the picker does.
  const loadDatasetBySource = async (source: string) => {
    const ds = datasetMeta.find(d => d.source === source)
    if (!ds) {
      setRestoringDataset(false)
      return false
    }
    setFetchingDataset(true)
    setDatasetLoadError(null)
    setFailedDatasetSource(source)
    try {
      const res = await retry(async () => {
        const response = await fetch(api.datasetCsvUrl(ds.source))
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response
      })
      handleLoad(new File([await res.blob()], ds.name + '.csv', { type: 'text/csv' }), ds.source)
      return true
    } catch (caught) {
      console.error('Failed to load dataset:', caught)
      setDatasetLoadError(`Could not load ${ds.name}. ${errorText(caught)}`)
      return false
    } finally {
      setFetchingDataset(false)
      setRestoringDataset(false)
    }
  }

  // Restore the most recently opened server dataset after the catalogue has loaded. The
  // catalogue and CSV requests both retry transient failures, and a missing/deleted source
  // is forgotten only after a successful catalogue response proves it is gone.
  useEffect(() => {
    if (!catalogReady || restoreAttempted.current) return
    restoreAttempted.current = true
    if (window.location.hash.startsWith('#compare=') || window.location.hash.startsWith('#guide=')) {
      setRestoringDataset(false)
      return
    }
    const source = readLastDatasetSource()
    const dataset = rememberedDataset(datasetMeta, source)
    if (!source || !dataset) {
      if (source) forgetDatasetSource(source)
      setRestoringDataset(false)
      return
    }
    void loadDatasetBySource(dataset.source)
  }, [catalogReady, datasetMeta]) // eslint-disable-line react-hooks/exhaustive-deps

  const runOverlay = useRunOverlay(data, datasetSource, replaceData)
  const activeRun = useActiveRun()
  const [showTrain, setShowTrain] = useState(false)
  // A child dataset was just saved: offer to train on it, carrying over the overlaid run's settings.
  const [createdDataset, setCreatedDataset] = useState<DatasetMeta | null>(null)
  const [trainOn, setTrainOn] = useState<{ dataset: DatasetMeta; params: Partial<TrainParams>; runParams?: Record<string, string>; preset?: string; parent?: string } | null>(null)
  const trainBlocked = (hasDataset: boolean) => trainDisabledReason({
    hasDataset,
    mlflowReachable: activeRun.presets?.mlflow.reachable ?? false,
    workerAvailable: activeRun.presets?.backend.available ?? false,
    workerReason: activeRun.presets?.backend.reason,
    activeRunId: activeRun.active ? activeRun.run!.run_id : null,
    demo: demoMode,
  })
  const trainOnLoaded = trainBlocked(!!data && !!datasetSource)
  const trainOnCreated = async () => {
    if (!createdDataset) return
    let params: Partial<TrainParams> = {}
    let runParams: Record<string, string> | undefined
    let preset: string | undefined
    const parent = runOverlay.overlay?.runId
    if (parent) {
      try {
        const earlier = await api.getRun(parent)
        params = paramsFromRun(earlier.params)
        runParams = earlier.params
        preset = earlier.tags['magnidata.preset'] ?? earlier.params.preset
      } catch { /* start from the preset defaults */ }
    }
    await activeRun.loadPresets()
    setTrainOn({ dataset: createdDataset, params, runParams, preset, parent })
  }
  const exportRelabelQueue = async () => {
    if (!data) return
    const notes: Record<string, string> = {}
    try {
      for (const n of await api.getNotes()) {
        const name = n.image_path.split(/[/\\]/).pop() ?? ''
        notes[name] = notes[name] ? `${notes[name]} | ${n.note}` : n.note
      }
    } catch { /* export without notes */ }
    const blob = new Blob([buildRelabelCsv(liveFilteredRows, data, notes)], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${data.fileName.replace(/\.csv$/i, '')}_relabel_queue.csv`
    a.click()
    URL.revokeObjectURL(a.href)
  }
  const [launchedRunId, setLaunchedRunId] = useState<string | null>(null)
  const autoOverlaid = useRef(new Set<string>())
  const onLaunched = (runId: string) => {
    setLaunchedRunId(runId)
    void runOverlay.refreshRuns()
    setShowTrain(false)
    setTrainOn(null)
    setCreatedDataset(null)
    activeRun.track(runId)
    setSelectedRunId(runId)
    setTab('experiments')
    void activeRun.loadPresets()
  }
  const trainingBadge = activeRun.active && activeRun.run
    ? `Training · ${progressLabel(activeRun.run.status, activeRun.epochsDone, Number(activeRun.run.params.epochs ?? 0), activeRun.run.tags)}`
    : null

  // Overlay the launched (or newest running) run by itself as soon as it has a split or losses to show.
  useEffect(() => {
    if (!data || runOverlay.overlay || runOverlay.compare || runOverlay.loading) return
    const target = autoOverlayTarget(pickerEntries(runOverlay.runs), launchedRunId, autoOverlaid.current)
    if (!target) return
    autoOverlaid.current.add(target.run.run_id)
    activeRun.toast('neutral', `Overlaying run ${target.run.name || target.run.run_id.slice(0, 8)}`)
    void runOverlay.select(target.run.run_id, 'latest')
  }, [runOverlay.runs, runOverlay.overlay, runOverlay.compare, runOverlay.loading, launchedRunId, data]) // eslint-disable-line react-hooks/exhaustive-deps

  // Live axes: while the overlaid run trains, pull in its new epochs as they are scored.
  const scoredCount = activeRun.run?.scored_epochs.length ?? 0
  useEffect(() => {
    if (scoredCount > 0) void runOverlay.refreshRuns()
    const ov = runOverlay.overlay
    if (activeRun.run && ov && ov.runId === activeRun.run.run_id && ov.epoch === 'latest' && scoredCount > 0) {
      void runOverlay.select(ov.runId, 'latest')
    }
  }, [scoredCount]) // eslint-disable-line react-hooks/exhaustive-deps

  // "Overlay on dataset" from the runs table: load the run's dataset first if it is not the open one.
  const compareOnDataset = (a: string, b: string, source: string) => {
    if (data && datasetSource === source) {
      void runOverlay.selectCompare(a, b)
      setTab('visual-explorer')
    } else {
      setPendingOverlay({ runId: a, compareWith: b, source })
      void loadDatasetBySource(source)
    }
  }

  const overlayRun = (runId: string, source: string) => {
    if (data && datasetSource === source) {
      void runOverlay.select(runId)
      setTab('visual-explorer')
    } else {
      setPendingOverlay({ runId, source })
      void loadDatasetBySource(source)
    }
  }
  useEffect(() => {
    if (pendingOverlay && data && datasetSource === pendingOverlay.source) {
      if (pendingOverlay.compareWith) void runOverlay.selectCompare(pendingOverlay.runId, pendingOverlay.compareWith)
      else void runOverlay.select(pendingOverlay.runId)
      setPendingOverlay(null)
      setTab('visual-explorer')
    }
  }, [pendingOverlay, data, datasetSource]) // eslint-disable-line react-hooks/exhaustive-deps

  // Experiments are global (MLflow), so they open without a dataset too.
  const goToExperiments = () => {
    setExperimentsDataset(null)
    clear()
    setDatasetSource(null)
    setTab('visual-explorer')
    setLandingView('experiments')
  }

  // Same view, narrowed to the runs trained on one dataset (the workspace panel's action).
  const goToDatasetExperiments = (source: string) => {
    goToExperiments()
    setExperimentsDataset(source)
  }

  // Persist the curated set (everything except the removed images) as a new child dataset.
  // The original dataset is untouched; on success the removals are cleared (they've been applied).
  const saveCurated = useCallback(async () => {
    if (!datasetSource || !data?.imagePath || !removalName.trim()) return
    const kept = data.rows.filter(r => !removed.has(r._idx as number))
    setSavingRemoval(true); setRemovalMsg(null)
    try {
      const image_names = kept
        .map(r => String(r[data.imagePath!] ?? ''))
        .filter(Boolean)
        .map(p => p.split(/[/\\]/).pop() as string)
      const created = await api.createDataset({
        name: removalName.trim(),
        description: `${kept.length} rows — ${removed.size} removed from ${data.fileName.replace(/\.csv$/i, '')}`,
        parent_source: datasetSource,
        image_names,
      })
      reloadDatasets()
      setCreatedDataset(created)
      setRemoved(new Set()); setShowRemovalSave(false); setRemovalName('')
    } catch (e) {
      setRemovalMsg(`error: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setSavingRemoval(false)
    }
  }, [datasetSource, data, removalName, removed, reloadDatasets])

  // Create a brand-new dataset from the hand-picked basket. Original dataset untouched; the basket
  // is cleared on success.
  const saveAdded = useCallback(async () => {
    if (!datasetSource || !data?.imagePath || !addName.trim() || !added.size) return
    const picked = data.rows.filter(r => added.has(r._idx as number))
    setSavingAdd(true); setAddMsg(null)
    try {
      const image_names = picked
        .map(r => String(r[data.imagePath!] ?? ''))
        .filter(Boolean)
        .map(p => p.split(/[/\\]/).pop() as string)
      const created = await api.createDataset({
        name: addName.trim(),
        description: `${picked.length} curated images from ${data.fileName.replace(/\.csv$/i, '')}`,
        parent_source: datasetSource,
        image_names,
      })
      reloadDatasets()
      setCreatedDataset(created)
      setAdded(new Set()); setShowAddSave(false); setAddName('')
    } catch (e) {
      setAddMsg(`error: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setSavingAdd(false)
    }
  }, [datasetSource, data, addName, added, reloadDatasets])

  const createComplement = useCallback(async () => {
    if (!datasetSource || !data?.imagePath || !complementName.trim() || !liveFilteredRows.length) return
    const basename = (row: CSVRow) => String(row[data.imagePath!] ?? '').split(/[/\\]/).pop() ?? ''
    const kept = data.rows.filter(row => !removed.has(row._idx as number)).map(basename).filter(Boolean)
    const query = liveFilteredRows.map(basename).filter(Boolean)
    setSavingComplement(true); setComplementMsg(null)
    try {
      const created = await api.createDataset({
        name: complementName.trim(),
        description: `${kept.length} retained samples plus up to ${complementTopK} feature-similar Cloud samples`,
        parent_source: datasetSource,
        image_names: kept,
        query_image_names: query,
        complement_top_k: complementTopK,
        cloud_source: CLOUD_DATASET_SOURCE,
      })
      await reloadDatasets()
      setCreatedDataset(created)
      setRemoved(new Set())
      setComplementName('')
      setComplementMsg(`Created ${created.row_count?.toLocaleString() ?? ''} sample dataset.`)
    } catch (error) {
      setComplementMsg(`error: ${errorText(error)}`)
    } finally {
      setSavingComplement(false)
    }
  }, [datasetSource, data, complementName, liveFilteredRows, removed, complementTopK, reloadDatasets])

  // Register the rows currently shown (filters, brushes and staged removals applied) as a child
  // dataset, so the selection appears in the workspace lineage graph.
  const saveSelection = useCallback(async () => {
    if (!datasetSource || !data?.imagePath || !selectionName.trim() || !liveFilteredRows.length) return
    setSavingSelection(true); setSelectionMsg(null); setSavedSelection(null)
    try {
      const image_names = liveFilteredRows
        .map(row => String(row[data.imagePath!] ?? ''))
        .filter(Boolean)
        .map(path => path.split(/[/\\]/).pop() as string)
      const created = await api.createDataset({
        name: selectionName.trim(),
        description: `${liveFilteredRows.length.toLocaleString()} selected samples from ${data.fileName.replace(/\.csv$/i, '')}`,
        parent_source: datasetSource,
        image_names,
      })
      await reloadDatasets()
      setSavedSelection(created)
      setSelectionName('')
    } catch (error) {
      setSelectionMsg(`error: ${errorText(error)}`)
    } finally {
      setSavingSelection(false)
    }
  }, [datasetSource, data, selectionName, liveFilteredRows, reloadDatasets])

  const activeSection: AppSection = data
    ? (tab === 'experiments' ? 'experiments' : 'explorer')
    : tab === 'schema'
      ? 'schema'
      : landingView === 'experiments'
        ? 'experiments'
        : landingView === 'help'
          ? 'help'
          : 'workspace'

  const pageCopy: Record<AppSection, [string, string]> = {
    workspace: ['Workspace', 'Browse dataset lineage, inspect a dataset, and jump into exploration or experiments'],
    explorer: ['Visual Explorer', data ? `Explore ${data.fileName} across features, embeddings, and samples` : 'Explore high-dimensional data and inspect samples'],
    experiments: ['Experiments', 'Track, compare, and analyze model training runs'],
    schema: ['Data Schema', 'Understand the fields available to every visualization'],
    help: ['User Guide', 'A guided path from AgriStress-500 to a curated, retrained dataset'],
  }
  const datasetPageCopy: Partial<Record<TabId, [string, string]>> = {
    'visual-explorer': ['Visual Explorer', 'Explore relationships, embeddings, and individual samples'],
    charts: ['Charts', 'Compare distributions across the current selection'],
    schema: ['Data Schema', 'Review and manage the fields in this dataset'],
    table: ['Data Table', 'Inspect the current selection row by row'],
    overview: ['Dataset Overview', 'Review summary statistics for the current selection'],
  }
  const headerCopy = data && datasetPageCopy[tab] ? datasetPageCopy[tab]! : pageCopy[activeSection]

  return (
    <DemoContext.Provider value={demoReason}>
    <div className="app-shell">
      <AppNavigation
        active={activeSection}
        onWorkspace={goHome}
        onExplorer={goToExplorer}
        onExperiments={goToExperiments}
        experimentsDisabled={demoReason}
        onSchema={goToSchema}
        onHelp={goToHelp}
        hasDataset={!!data}
        activeTab={tab}
        onDatasetView={setTab}
      />

      <div className="app-stage">
      <Header
        pageTitle={headerCopy[0]}
        pageSubtitle={headerCopy[1]}
        data={data}
        filteredCount={liveFilteredRows.length}
        buildJob={buildJob.job}
        onDismissBuildJob={buildJob.dismiss}
        trainingBadge={trainingBadge}
      />

      {/* Main content */}
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>

        {!data && !loading && !restoringDataset && !fetchingDataset && !datasetLoadError && !error && tab !== 'schema' && landingView === 'landing' && (
          <WorkspaceView
            datasets={datasetMeta}
            onOpenDataset={source => void loadDatasetBySource(source)}
            onViewExperiments={goToDatasetExperiments}
            onDelete={deleteDataset}
            onOpenByod={() => setShowByod(true)}
            demoReason={demoReason}
            onUploadFile={file => handleLoad(file)}
            onDatasetsChanged={reloadDatasets}
          />
        )}

        {!data && !loading && !restoringDataset && !fetchingDataset && !datasetLoadError && !error && tab !== 'schema' && landingView === 'help' && (
          <GuidePage onOpenByod={demoMode ? undefined : () => setShowByod(true)} />
        )}

        {!data && !loading && !restoringDataset && !fetchingDataset && !datasetLoadError && !error && tab !== 'schema' && landingView === 'experiments' && (
          <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <ExperimentsTab key={experimentsDataset ?? 'all'} datasetSource={null} datasetFilter={experimentsDataset} selectedRunId={selectedRunId} onSelectRun={r => setSelectedRunId(r.run_id)}
              onCloseRun={() => setSelectedRunId(null)} knownDatasets={datasetMeta.map(d => d.source)} onLoadDataset={loadDatasetBySource} onOverlayRun={overlayRun} onCompareOnDataset={compareOnDataset} onSelectRunId={setSelectedRunId}
                trainDisabled={trainBlocked(!!data && !!datasetSource)} workerReason={activeRun.presets && !activeRun.presets.backend.available ? (activeRun.presets.backend.reason ?? 'no trainer worker has reported in') : null}
                onTrain={() => { void activeRun.loadPresets(); setShowTrain(true) }} onRunChanged={() => void activeRun.refresh()}
                liveKey={`${activeRun.run?.run_id}:${activeRun.run?.status}:${activeRun.epochsDone}`} />
          </div>
        )}

        {(loading || restoringDataset || fetchingDataset) && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="text-center">
              <div className="loading-spinner" />
              <p className="text-sm mt-3" style={{ color: 'var(--c-t3)' }}>
                {restoringDataset ? 'Restoring your last dataset…' : fetchingDataset ? 'Loading dataset…' : 'Parsing CSV…'}
              </p>
            </div>
          </div>
        )}

        {!data && datasetLoadError && !loading && !restoringDataset && !fetchingDataset && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="text-center p-8" style={{ maxWidth: 560 }}>
              <p className="text-4xl mb-3">⚠</p>
              <p className="font-semibold mb-1" style={{ color: 'var(--c-t1)' }}>Dataset could not be restored</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--c-t3)', marginBottom: 14 }}>{datasetLoadError}</p>
              <div style={{ display: 'flex', justifyContent: 'center', gap: 8 }}>
                <button className="btn-primary" onClick={() => {
                  setDatasetLoadError(null)
                  if (failedDatasetSource) void loadDatasetBySource(failedDatasetSource)
                  else void reloadDatasets()
                }}>Retry</button>
                <button className="btn-ghost" onClick={() => {
                  setDatasetLoadError(null)
                  setFailedDatasetSource(null)
                  setLandingView('landing')
                }}>Choose another dataset</button>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="text-center p-8">
              <p className="text-4xl mb-3">⚠</p>
              <p className="font-semibold mb-1" style={{ color: 'var(--c-t1)' }}>Failed to parse CSV</p>
              <p className="text-sm font-semibold" style={{ color: 'var(--c-t3)' }}>{error}</p>
            </div>
          </div>
        )}

        {/* Tab bar (full width) + sidebar/content row below it.
            Schema is always available — doesn't need a loaded file. */}
        {(data || (!data && !loading && tab === 'schema')) && (
          <>
            {/* Tab bar — spans the full width above the sidebar */}
            <div className="tab-bar shrink-0" style={{ margin: '0', flexShrink: 0 }}>
              <div className="tab-bar__nav">
                {TABS.map(t => (
                  <button
                    key={t.id}
                    onClick={() => setTab(t.id)}
                    className="tab-item"
                    data-active={tab === t.id}
                    style={{ display: 'flex', alignItems: 'center', gap: 6 }}
                  >
                    {t.icon}
                    {t.label}
                  </button>
                ))}
              </div>

              {data && <div className="tab-bar__tools" aria-label="Dataset tools">
              {data && (
                <button className="btn-primary" disabled={!!trainOnLoaded} title={trainOnLoaded ?? 'Train a model on this dataset'}
                        onClick={() => { void activeRun.loadPresets(); setShowTrain(true) }}
                        style={{ fontSize: 11, padding: '3px 10px', marginRight: 10, opacity: trainOnLoaded ? 0.5 : 1, cursor: demoReason ? 'not-allowed' : undefined }}>Train</button>
              )}
              {data && !!data.runColumns?.length && (
                <button className="btn-ghost" onClick={() => void exportRelabelQueue()}
                        title="Download the selected images with their training losses and notes, hardest first"
                        style={{ fontSize: 11, padding: '3px 10px', marginRight: 10, whiteSpace: 'nowrap' }}>Export relabel queue</button>
              )}
              {data && datasetSource && (
                <RunOverlayPicker runs={runOverlay.runs} overlay={runOverlay.overlay} compare={runOverlay.compare} match={runOverlay.match}
                                  loading={runOverlay.loading} error={runOverlay.error}
                                  onSelect={id => void runOverlay.select(id)} onClear={runOverlay.clear} onOpen={() => void runOverlay.refreshRuns()} />
              )}

              {data && runOverlay.overlay && hasSplitColumn && (
                <span role="status" title="Rows of the loaded dataset in each split of the overlaid run"
                      style={{ fontSize: 11, marginRight: 10, whiteSpace: 'nowrap', color: 'var(--c-t2)' }}>
                  {splitSummary(data.rows)}
                </span>
              )}

              {data && runOverlay.overlay && (
                <EpochScrubber scoredEpochs={runOverlay.scoredEpochs} epoch={runOverlay.overlay.epoch} loading={runOverlay.loading}
                               onChange={e => void runOverlay.select(runOverlay.overlay!.runId, e)} />
              )}

              {/* Visual-explorer plot controls — moved up here from the explorer panel.
                  Vertical split, then Color by / Sort / Fields on the right. */}
              {data && tab === 'visual-explorer' && visibleData && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginRight: 10 }}>
                  <div style={{ width: 1, height: 22, background: 'var(--c-border)', marginRight: 3 }} />
                  <span style={{ fontSize: 11, color: 'var(--c-t3)' }}>Color by</span>
                  <select value={colorBy} onChange={e => chooseColorBy(e.target.value)}
                          style={{ fontSize: 11, borderRadius: 7, padding: '3px 8px', outline: 'none',
                                   background: 'var(--c-surface)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }}>
                    {[...visibleData.categoricalCols, ...visibleData.numericCols].filter(c => !visibleData.runColumns?.includes(c)).map(c => <option key={c} value={c}>{c}</option>)}
                    {!!visibleData.runColumns?.length && (
                      <optgroup label="Training">
                        {[...visibleData.categoricalCols, ...visibleData.numericCols].filter(c => visibleData.runColumns!.includes(c)).map(c => <option key={c} value={c}>{c}</option>)}
                      </optgroup>
                    )}
                  </select>
                  <span style={{ marginLeft: 4, fontSize: 11, color: 'var(--c-t3)' }}>Sort</span>
                  <select value={sortMode}
                          onChange={e => setSortMode(e.target.value as 'default' | 'variance' | 'corr')}
                          style={{ fontSize: 11, borderRadius: 7, padding: '3px 8px', outline: 'none',
                                   background: 'var(--c-surface)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }}>
                    <option value="default">Default</option>
                    <option value="variance">Variance</option>
                    <option value="corr">|Corr|</option>
                  </select>
                  <button onClick={() => setShowFields(v => !v)}
                          style={{ display: 'flex', alignItems: 'center', gap: 5, marginLeft: 4,
                                   fontSize: 11, padding: '3px 10px', borderRadius: 7, cursor: 'pointer',
                                   background: showFields ? 'var(--pai-navy)' : 'var(--c-surface)',
                                   color:      showFields ? '#fff' : 'var(--c-t1)',
                                   border: `1px solid ${showFields ? 'var(--pai-navy)' : 'var(--c-border)'}` }}>
                    <Columns size={12} /> Fields
                  </button>
                  <div style={{ width: 1, height: 22, background: 'var(--c-border)', marginLeft: 3 }} />
                </div>
              )}

              {data && activeCount > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 8, fontSize: 12, color: 'var(--c-t3)' }}>
                  <span style={{ fontWeight: 600, color: 'var(--c-accent)' }}>{liveFilteredRows.length.toLocaleString()}</span>
                  <span>of {data.rows.length.toLocaleString()} rows</span>
                </div>
              )}

              {/* Pending image removals — save the curated set as a new dataset, or discard */}
              {data && removed.size > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 10 }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontWeight: 600,
                                 padding: '3px 9px', borderRadius: 20,
                                 background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.35)', color: '#dc2626' }}>
                    <Trash2 size={11} /> {removed.size.toLocaleString()} removed
                  </span>
                  {datasetSource && data.imagePath && (
                    <button onClick={() => { setShowRemovalSave(v => !v); setRemovalMsg(null) }}
                            title="Save the remaining images as a new dataset"
                            style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontWeight: 600,
                                     padding: '3px 9px', borderRadius: 7, cursor: 'pointer',
                                     background: showRemovalSave ? 'var(--pai-navy)' : 'var(--c-accent)', color: '#fff',
                                     border: '1px solid var(--c-accent)' }}>
                      <Save size={11} /> Save dataset
                    </button>
                  )}
                  <button onClick={discardRemovals} title="Discard all pending removals"
                          style={{ fontSize: 11, fontWeight: 600, padding: '3px 9px', borderRadius: 7, cursor: 'pointer',
                                   background: 'var(--c-surface)', color: 'var(--c-t2)', border: '1px solid var(--c-border)' }}>
                    Discard
                  </button>
                </div>
              )}

              {/* Curation basket — hand-picked images → save as a brand-new dataset */}
              {data && added.size > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 10 }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontWeight: 600,
                                 padding: '3px 9px', borderRadius: 20,
                                 background: 'rgba(41,204,165,0.14)', border: '1px solid rgba(41,204,165,0.4)', color: '#0a8f6e' }}>
                    <Plus size={11} /> {added.size.toLocaleString()} picked
                  </span>
                  <button onClick={exportAdded}
                          title="Download the picked images as a CSV"
                          style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontWeight: 600,
                                   padding: '3px 9px', borderRadius: 7, cursor: 'pointer',
                                   background: 'var(--c-surface)', color: 'var(--c-t1)', border: '1px solid var(--c-border)' }}>
                    <Download size={11} /> Export
                  </button>
                  {datasetSource && data.imagePath && (
                    <button onClick={() => { setShowAddSave(v => !v); setAddMsg(null) }}
                            title="Create a new dataset from the picked images"
                            style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontWeight: 600,
                                     padding: '3px 9px', borderRadius: 7, cursor: 'pointer',
                                     background: showAddSave ? 'var(--pai-navy)' : 'var(--c-accent)', color: '#fff',
                                     border: '1px solid var(--c-accent)' }}>
                      <Save size={11} /> Create dataset
                    </button>
                  )}
                  <button onClick={clearAdded} title="Clear the curation basket"
                          style={{ fontSize: 11, fontWeight: 600, padding: '3px 9px', borderRadius: 7, cursor: 'pointer',
                                   background: 'var(--c-surface)', color: 'var(--c-t2)', border: '1px solid var(--c-border)' }}>
                    Clear
                  </button>
                </div>
              )}
              </div>}
            </div>

            {/* Inline save form for the curated (post-removal) dataset */}
            {data && removed.size > 0 && showRemovalSave && datasetSource && data.imagePath && (
              <div style={{ flexShrink: 0, borderBottom: '1px solid var(--c-border)', background: 'var(--c-surface)',
                            padding: '8px 14px', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                <span style={{ fontSize: 11, color: 'var(--c-t2)', fontWeight: 600 }}>
                  New dataset with {(data.rows.length - removed.size).toLocaleString()} images
                  <span style={{ color: '#dc2626', marginLeft: 4 }}>· {removed.size} removed</span>
                </span>
                <input autoFocus placeholder="Dataset name" value={removalName}
                       onChange={e => setRemovalName(e.target.value)}
                       onKeyDown={e => { if (e.key === 'Enter') saveCurated() }}
                       style={{ fontSize: 11, padding: '4px 8px', borderRadius: 6, outline: 'none', width: 200,
                                background: 'var(--c-raised)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }} />
                <button onClick={saveCurated} disabled={!removalName.trim() || savingRemoval || !!demoReason} title={demoReason}
                        style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 12px', borderRadius: 6,
                                 fontSize: 11, fontWeight: 600, cursor: removalName.trim() && !savingRemoval && !demoReason ? 'pointer' : 'not-allowed',
                                 background: removalName.trim() && !savingRemoval && !demoReason ? 'var(--c-accent)' : 'var(--c-raised)',
                                 color: removalName.trim() && !savingRemoval && !demoReason ? '#fff' : 'var(--c-t3)',
                                 border: `1px solid ${removalName.trim() && !savingRemoval && !demoReason ? 'var(--c-accent)' : 'var(--c-border)'}` }}>
                  {savingRemoval ? <Loader2 size={12} style={{ animation: 'spin 0.7s linear infinite' }} /> : <Save size={12} />}
                  {savingRemoval ? 'Creating…' : 'Create dataset'}
                </button>
                <button onClick={() => setShowRemovalSave(false)} style={{ background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--c-t3)', padding: 2 }}><X size={14} /></button>
                {removalMsg && <span style={{ fontSize: 11, color: '#ef4444' }}>{removalMsg}</span>}
              </div>
            )}

            {data && createdDataset && (
              <div role="status" style={{ flexShrink: 0, borderBottom: '1px solid var(--c-border)', background: 'var(--c-surface)',
                                          padding: '8px 14px', display: 'flex', alignItems: 'center', gap: 10, fontSize: 11 }}>
                <span style={{ color: 'var(--c-t2)', fontWeight: 600 }}>
                  Dataset “{createdDataset.name}” created{createdDataset.row_count ? ` (${createdDataset.row_count.toLocaleString()} images)` : ''}.
                </span>
                <button className="btn-primary" disabled={!!trainBlocked(true)} title={trainBlocked(true) ?? 'Train on the new dataset, linked to the overlaid run'}
                        onClick={() => void trainOnCreated()} style={{ fontSize: 11, padding: '3px 10px', opacity: trainBlocked(true) ? 0.5 : 1 }}>Train on it</button>
                <button onClick={() => setCreatedDataset(null)} aria-label="Dismiss" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--c-t3)' }}><X size={14} /></button>
              </div>
            )}

            {/* Inline save form for the curation basket (hand-picked images → new dataset) */}
            {data && added.size > 0 && showAddSave && datasetSource && data.imagePath && (
              <div style={{ flexShrink: 0, borderBottom: '1px solid var(--c-border)', background: 'var(--c-surface)',
                            padding: '8px 14px', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                <span style={{ fontSize: 11, color: 'var(--c-t2)', fontWeight: 600 }}>
                  New dataset from {added.size.toLocaleString()} picked images
                </span>
                <input autoFocus placeholder="Dataset name" value={addName}
                       onChange={e => setAddName(e.target.value)}
                       onKeyDown={e => { if (e.key === 'Enter') saveAdded() }}
                       style={{ fontSize: 11, padding: '4px 8px', borderRadius: 6, outline: 'none', width: 200,
                                background: 'var(--c-raised)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }} />
                <button onClick={saveAdded} disabled={!addName.trim() || savingAdd || !!demoReason} title={demoReason}
                        style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 12px', borderRadius: 6,
                                 fontSize: 11, fontWeight: 600, cursor: addName.trim() && !savingAdd && !demoReason ? 'pointer' : 'not-allowed',
                                 background: addName.trim() && !savingAdd && !demoReason ? 'var(--c-accent)' : 'var(--c-raised)',
                                 color: addName.trim() && !savingAdd && !demoReason ? '#fff' : 'var(--c-t3)',
                                 border: `1px solid ${addName.trim() && !savingAdd && !demoReason ? 'var(--c-accent)' : 'var(--c-border)'}` }}>
                  {savingAdd ? <Loader2 size={12} style={{ animation: 'spin 0.7s linear infinite' }} /> : <Save size={12} />}
                  {savingAdd ? 'Creating…' : 'Create dataset'}
                </button>
                <button onClick={() => setShowAddSave(false)} style={{ background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--c-t3)', padding: 2 }}><X size={14} /></button>
                {addMsg && <span style={{ fontSize: 11, color: '#ef4444' }}>{addMsg}</span>}
              </div>
            )}

            {/* Row below the tab bar: sidebar (left) + tab content (right) */}
            <div className="dataset-workspace-row" style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
              {/* Sidebar — hidden on visual-explorer and schema tabs */}
              {data && tab !== 'visual-explorer' && tab !== 'schema' && tab !== 'experiments' && (
                <Sidebar
                  data={data}
                  filters={filters}
                  activeCount={activeCount}
                  onRange={setRange}
                  onCategory={setCategory}
                  onSearch={setSearch}
                  onReset={reset}
                />
              )}

              {/* Tab content */}
              <div style={{ flex: 1, minWidth: 0, minHeight: 0, overflow: 'hidden' }}>
                {!data && tab === 'schema' && (
                  <Schema data={null} hiddenCols={hiddenCols} onToggleCol={toggleCol} />
                )}
                {data && tab === 'overview' && (
                  <Overview data={visibleData!} rows={liveFilteredRows} />
                )}
                {data && tab === 'table' && (
                  <DataTable data={visibleData!} rows={liveFilteredRows}
                             onPlay={row => setPreview({ row, rows: liveFilteredRows })} />
                )}
                {data && tab === 'visual-explorer' && (
                  <ParallelCoords
                    data={visibleData!}
                    rows={data.rows}
                    embeddingsAvailable={embeddingsAvailable}
                    datasetSource={datasetSource}
                    onConstraintsChange={setPCConstraints}
                    onPlay={(row, rows) => setPreview({ row, rows })}
                    partitions={partitions}
                    onCommitClusterColumn={setClusterColumn}
                    removed={removed}
                    colorBy={colorBy} setColorBy={chooseColorBy}
                    sortMode={sortMode} setSortMode={setSortMode}
                    showFields={showFields} setShowFields={setShowFields}
                  />
                )}
                {data && tab === 'charts' && (
                  <Histograms data={visibleData!} rows={liveFilteredRows} />
                )}
                {data && tab === 'experiments' && (
                  <ExperimentsTab datasetSource={datasetSource} selectedRunId={selectedRunId}
                                  onSelectRun={r => setSelectedRunId(r.run_id)} onCloseRun={() => setSelectedRunId(null)}
                                  knownDatasets={datasetMeta.map(d => d.source)} onLoadDataset={loadDatasetBySource} onOverlayRun={overlayRun} onCompareOnDataset={compareOnDataset} onSelectRunId={setSelectedRunId}
                trainDisabled={trainBlocked(!!data && !!datasetSource)} workerReason={activeRun.presets && !activeRun.presets.backend.available ? (activeRun.presets.backend.reason ?? 'no trainer worker has reported in') : null}
                onTrain={() => { void activeRun.loadPresets(); setShowTrain(true) }} onRunChanged={() => void activeRun.refresh()}
                liveKey={`${activeRun.run?.run_id}:${activeRun.run?.status}:${activeRun.epochsDone}`} />
                )}
                {data && tab === 'schema' && (
                  <Schema data={data} hiddenCols={hiddenCols} onToggleCol={toggleCol} />
                )}
              </div>

              {data && tab === 'visual-explorer' && visibleData && (
                <aside className="explorer-config" data-open={explorerPanelOpen} aria-label="Visual Explorer configuration">
                  <div className="explorer-config__header">
                    {explorerPanelOpen && <span><SlidersHorizontal size={14} /> Explorer setup</span>}
                    <button className="btn-icon" onClick={() => setExplorerPanelOpen(open => !open)}
                            aria-label={explorerPanelOpen ? 'Collapse explorer setup' : 'Expand explorer setup'}>
                      {explorerPanelOpen ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
                    </button>
                  </div>

                  {explorerPanelOpen && (
                    <div className="explorer-config__body">
                      <section className="explorer-config__section">
                        <div className="explorer-config__section-title"><span>Workflow</span><small>Train and export</small></div>
                        <button className="btn-primary explorer-config__wide" disabled={!!trainOnLoaded}
                                title={trainOnLoaded ?? 'Train a model on this dataset'}
                                onClick={() => { void activeRun.loadPresets(); setShowTrain(true) }}
                                style={{ opacity: trainOnLoaded ? 0.5 : 1, cursor: demoReason ? 'not-allowed' : undefined }}>
                          <FlaskConical size={14} /> Train model
                        </button>
                        <button className="btn-ghost explorer-config__wide" disabled={exportingSelection || !liveFilteredRows.length}
                                onClick={() => void exportSelection()}>
                          {exportingSelection ? <Loader2 size={13} style={{ animation: 'spin .7s linear infinite' }} /> : <Download size={13} />}
                          Export selection
                        </button>
                        {datasetSource && data.imagePath && (
                          <div className="explorer-config__save">
                            <label className="explorer-config__field">
                              <span>Save selection as a dataset</span>
                              <input value={selectionName} placeholder="Name for the new dataset"
                                     onChange={event => setSelectionName(event.target.value)}
                                     onKeyDown={event => { if (event.key === 'Enter') void saveSelection() }} />
                            </label>
                            <button className="btn-primary explorer-config__wide"
                                    disabled={savingSelection || !selectionName.trim() || !liveFilteredRows.length || !!demoReason}
                                    title={demoReason}
                                    style={demoReason ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}
                                    onClick={() => void saveSelection()}>
                              {savingSelection ? <Loader2 size={13} className="spin" /> : <Database size={13} />}
                              Save {liveFilteredRows.length.toLocaleString()} samples as dataset
                            </button>
                            {selectionMsg && <p className="explorer-config__hint" role="alert">{selectionMsg}</p>}
                            {savedSelection && (
                              <p className="explorer-config__hint" role="status">
                                Created “{savedSelection.name}”.{' '}
                                <button className="explorer-config__link" onClick={() => { rememberWorkspaceSelection(savedSelection.source); goHome() }}>
                                  View in workspace
                                </button>
                              </p>
                            )}
                          </div>
                        )}
                        {!!data.runColumns?.length && (
                          <button className="btn-ghost explorer-config__wide" onClick={() => void exportRelabelQueue()}>
                            <Download size={13} /> Export relabel queue
                          </button>
                        )}
                      </section>

                      {datasetSource && data.imagePath && (
                        <section className="explorer-config__section">
                          <div className="explorer-config__section-title"><span>Complement</span><small>Expand from Cloud</small></div>
                          <p className="explorer-config__hint">
                            Find Cloud samples closest to the {liveFilteredRows.length.toLocaleString()} currently filtered samples.
                            {removed.size > 0 ? ` The ${removed.size.toLocaleString()} staged removals will be excluded.` : ''}
                          </p>
                          <label className="explorer-config__field">
                            <span>New dataset name</span>
                            <input value={complementName} placeholder="Curated + complement"
                                   onChange={event => setComplementName(event.target.value)} />
                          </label>
                          <label className="explorer-config__field">
                            <span>Top-K Cloud samples</span>
                            <input type="number" min={1} max={5000} value={complementTopK}
                                   onChange={event => setComplementTopK(Math.max(1, Math.min(5000, Number(event.target.value) || 1)))} />
                          </label>
                          <button className="btn-primary explorer-config__wide"
                                  disabled={savingComplement || !complementName.trim() || !liveFilteredRows.length || datasetSource === CLOUD_DATASET_SOURCE || !!demoReason}
                                  title={demoReason}
                                  style={demoReason ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}
                                  onClick={() => void createComplement()}>
                            {savingComplement ? <Loader2 size={13} className="spin" /> : <CloudDownload size={13} />}
                            {savingComplement ? 'Finding complement…' : 'Create complemented dataset'}
                          </button>
                          {datasetSource === CLOUD_DATASET_SOURCE && <p className="explorer-config__hint">Cloud Dataset already contains the full complement pool.</p>}
                          {complementMsg && <p className="explorer-config__hint" role="status">{complementMsg}</p>}
                        </section>
                      )}

                      {datasetSource && (
                        <section className="explorer-config__section">
                          <div className="explorer-config__section-title"><span>Experiment overlay</span><small>Add training signals</small></div>
                          <RunOverlayPicker runs={runOverlay.runs} overlay={runOverlay.overlay} compare={runOverlay.compare} match={runOverlay.match}
                                            loading={runOverlay.loading} error={runOverlay.error}
                                            onSelect={id => void runOverlay.select(id)} onClear={runOverlay.clear}
                                            onOpen={() => void runOverlay.refreshRuns()} />
                          {runOverlay.overlay && hasSplitColumn && <p className="explorer-config__hint">{splitSummary(data.rows)}</p>}
                          {runOverlay.overlay && (
                            <EpochScrubber scoredEpochs={runOverlay.scoredEpochs} epoch={runOverlay.overlay.epoch}
                                           loading={runOverlay.loading} onChange={epoch => void runOverlay.select(runOverlay.overlay!.runId, epoch)} />
                          )}
                        </section>
                      )}

                      <section className="explorer-config__section">
                        <div className="explorer-config__section-title"><span>Appearance</span><small>Shape the canvas</small></div>
                        <label className="explorer-config__field">
                          <span>Color by</span>
                          <select value={colorBy} onChange={event => chooseColorBy(event.target.value)}>
                            {[...visibleData.categoricalCols, ...visibleData.numericCols].filter(col => !visibleData.runColumns?.includes(col)).map(col => <option key={col} value={col}>{col}</option>)}
                            {!!visibleData.runColumns?.length && (
                              <optgroup label="Training">
                                {[...visibleData.categoricalCols, ...visibleData.numericCols].filter(col => visibleData.runColumns!.includes(col)).map(col => <option key={col} value={col}>{col}</option>)}
                              </optgroup>
                            )}
                          </select>
                        </label>
                        <label className="explorer-config__field">
                          <span>Axis order</span>
                          <select value={sortMode} onChange={event => setSortMode(event.target.value as 'default' | 'variance' | 'corr')}>
                            <option value="default">Dataset order</option>
                            <option value="variance">Highest variance</option>
                            <option value="corr">Strongest correlation</option>
                          </select>
                        </label>
                        <button className="btn-ghost explorer-config__wide" data-active={showFields}
                                onClick={() => setShowFields(open => !open)}>
                          <Columns size={13} /> {showFields ? 'Close field selection' : 'Choose visible fields'}
                        </button>
                      </section>

                      {(activeCount > 0 || removed.size > 0 || added.size > 0) && (
                        <section className="explorer-config__section">
                          <div className="explorer-config__section-title"><span>Current selection</span><small>Curation actions</small></div>
                          {activeCount > 0 && <p className="explorer-config__metric"><strong>{liveFilteredRows.length.toLocaleString()}</strong> of {data.rows.length.toLocaleString()} samples</p>}
                          {removed.size > 0 && (
                            <div className="explorer-config__action-row">
                              <span className="explorer-config__badge explorer-config__badge--danger"><Trash2 size={11} /> {removed.size} removed</span>
                              {datasetSource && data.imagePath && <button onClick={() => { setShowRemovalSave(true); setRemovalMsg(null) }}>Save</button>}
                              <button onClick={discardRemovals}>Discard</button>
                            </div>
                          )}
                          {added.size > 0 && (
                            <div className="explorer-config__action-row">
                              <span className="explorer-config__badge"><Plus size={11} /> {added.size} picked</span>
                              <button onClick={exportAdded}>Export</button>
                              {datasetSource && data.imagePath && <button onClick={() => { setShowAddSave(true); setAddMsg(null) }}>Create</button>}
                              <button onClick={clearAdded}>Clear</button>
                            </div>
                          )}
                        </section>
                      )}
                    </div>
                  )}
                </aside>
              )}
            </div>
          </>
        )}
      </div>
      </div>

      {trainOn && activeRun.presets && (
        <TrainDialog key={trainOn.dataset.source} presets={activeRun.presets} datasetSource={trainOn.dataset.source}
                     datasetName={trainOn.dataset.name} parentRunId={trainOn.parent} initialParams={trainOn.params} initialRunParams={trainOn.runParams}
                     initialPreset={trainOn.preset} onClose={() => setTrainOn(null)} onLaunched={onLaunched} />
      )}

      {showTrain && activeRun.presets && data && datasetSource && (
        <TrainDialog presets={activeRun.presets} datasetSource={datasetSource} datasetName={data.fileName}
                     onClose={() => setShowTrain(false)} onLaunched={onLaunched} />
      )}

      <div aria-live="polite" style={{ position: 'fixed', right: 16, bottom: 16, zIndex: 60, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {activeRun.toasts.map(t => (
          <div key={t.id} role="status" className="flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-semibold"
               style={{ background: t.tone === 'success' ? 'var(--color-success-bg)' : t.tone === 'danger' ? 'var(--color-danger-bg)' : 'var(--c-raised)',
                        color: t.tone === 'success' ? 'var(--color-success-text)' : t.tone === 'danger' ? 'var(--color-danger)' : 'var(--c-t2)',
                        border: '1px solid var(--c-border)', maxWidth: 380 }}>
            {t.message}
            <button onClick={() => activeRun.dismissToast(t.id)} aria-label="Dismiss" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'inherit' }}>×</button>
          </div>
        ))}
      </div>

      {showByod && (
        <ByodDialog
          datasets={datasetMeta}
          buildJob={buildJob.job}
          buildBusy={buildJob.busy}
          onClose={() => setShowByod(false)}
          startUpload={buildJob.startUpload}
          startDemo={buildJob.startDemo}
          startLocal={buildJob.startLocal}
        />
      )}

      {/* Image Modal — navigates only over the selection captured when it was opened */}
      {data && preview && (
        <ImageModal
          data={data}
          row={preview.row}
          allRows={preview.rows}
          onClose={() => setPreview(null)}
          partitionMap={partitions.map}
          removed={removed}
          onToggleRemove={toggleRemove}
          added={added}
          onToggleAdd={toggleAdd}
          run={runOverlay.overlay ? { runId: runOverlay.overlay.runId, epoch: runOverlay.overlay.epoch, scoredEpochs: runOverlay.scoredEpochs } : undefined}
        />
      )}
    </div>
    </DemoContext.Provider>
  )
}
