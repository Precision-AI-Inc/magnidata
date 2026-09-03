import { useState, useEffect, useMemo, useCallback } from 'react'
import { BarChart2, Table2, Share2, Activity, LayoutGrid, Trash2, Save, X, Loader2, Plus, Download, Columns } from 'lucide-react'
import { api } from './api'
import type { DatasetMeta } from './api'
import { useCSV } from './hooks/useCSV'
import { useFilters } from './hooks/useFilters'
import { usePartitions } from './hooks/usePartitions'
import { useBuildJob } from './hooks/useBuildJob'
import type { CSVRow, TabId } from './types'

import { Header }         from './components/Header'
import { Sidebar }        from './components/Sidebar'
import { Landing }        from './components/Landing'
import { CSVLoader }      from './components/CSVLoader'
import { ByodDialog }     from './components/ByodDialog'
import { HelpPage }       from './components/HelpPage'
import { DataTable }      from './components/DataTable'
import { Overview }       from './components/Overview'
import { ParallelCoords } from './components/ParallelCoords'
import { Histograms }     from './components/Histograms'
import { ImageModal }     from './components/ImageModal'
import { Schema }         from './components/Dictionary'

type LandingView = 'landing' | 'datasets' | 'help'

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'visual-explorer', label: 'Visual Explorer', icon: <Share2 size={13} /> },
  { id: 'charts',          label: 'Charts',          icon: <BarChart2 size={13} /> },
  { id: 'schema',          label: 'Schema',          icon: <LayoutGrid size={13} /> },
  { id: 'table',           label: 'Table',           icon: <Table2 size={13} /> },
  { id: 'overview',        label: 'Overview',        icon: <Activity size={13} /> },
]

export default function App() {
  const [tab, setTab]           = useState<TabId>('visual-explorer')
  // Image preview holds the clicked row + the exact set of rows to navigate (the current selection)
  const [preview, setPreview]   = useState<{ row: CSVRow; rows: CSVRow[] } | null>(null)
  const [hiddenCols, setHiddenCols] = useState<Set<string>>(new Set())
  const [datasetSource, setDatasetSource] = useState<string | null>(null)
  const [datasetMeta, setDatasetMeta] = useState<DatasetMeta[]>([])
  const [landingView, setLandingView] = useState<LandingView>('landing')
  const [showByod, setShowByod] = useState(false)

  // Staged image removals (by row _idx). They only shape the live views in memory; the original
  // dataset is never touched. They "apply" only when saved as a new dataset (or discarded otherwise).
  const [removed, setRemoved] = useState<Set<number>>(new Set())
  const [showRemovalSave, setShowRemovalSave] = useState(false)
  const [removalName, setRemovalName] = useState('')
  const [savingRemoval, setSavingRemoval] = useState(false)
  const [removalMsg, setRemovalMsg] = useState<string | null>(null)

  // Hand-picked images (a curation "basket"). Saving turns the picked set into a brand-new dataset.
  // Visual-explorer plot controls — lifted up so they render in the tab bar (see below).
  const [colorBy, setColorBy]       = useState('')
  const [sortMode, setSortMode]     = useState<'default' | 'variance' | 'corr'>('default')
  const [showFields, setShowFields] = useState(false)

  const [added, setAdded] = useState<Set<number>>(new Set())
  const [showAddSave, setShowAddSave] = useState(false)
  const [addName, setAddName] = useState('')
  const [savingAdd, setSavingAdd] = useState(false)
  const [addMsg, setAddMsg] = useState<string | null>(null)

  // Embeddings now live entirely on the API — the browser only needs to know they exist.
  const embeddingsAvailable = useMemo(
    () => !!datasetSource && !!datasetMeta.find(d => d.source === datasetSource)?.has_embeddings,
    [datasetSource, datasetMeta],
  )

  const { data, loading, error, load, setClusterColumn, clear } = useCSV()
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

  // Filtered rows with the staged removals excluded — what the tabs/preview show. The full row set
  // (data.rows) is kept for the explorer so the server-side embedding projection stays aligned.
  const liveFilteredRows = useMemo(
    () => (removed.size ? filteredRows.filter(r => !removed.has(r._idx as number)) : filteredRows),
    [filteredRows, removed],
  )

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

  const reloadDatasets = useCallback(() => {
    api.datasets().then(setDatasetMeta).catch(() => {})
  }, [])

  const deleteDataset = useCallback((id: number) => {
    api.deleteDataset(id).then(reloadDatasets).catch(() => {})
  }, [reloadDatasets])

  // Global dataset-build progress — lives here (not inside the BYOD dialog) so it keeps
  // polling and stays visible (as a Header badge) after the dialog that started it closes.
  const handleBuildDone = useCallback((_dataset: DatasetMeta) => {
    reloadDatasets()
    setLandingView('datasets')     // the finished dataset shows up in the picker grid
  }, [reloadDatasets])
  const buildJob = useBuildJob(handleBuildDone)

  // Load the dataset catalogue (for embeddings availability) on mount
  useEffect(() => {
    reloadDatasets()
  }, [reloadDatasets])

  const handleLoad = (f: File, source?: string) => {
    load(f)
    setTab('visual-explorer')
    setPreview(null)
    reset()
    setHiddenCols(new Set())
    setRemoved(new Set())          // pending removals don't carry across loads
    setShowRemovalSave(false); setRemovalName(''); setRemovalMsg(null)
    setAdded(new Set())            // the curation basket is per-dataset
    setShowAddSave(false); setAddName(''); setAddMsg(null)
    // Embeddings are computed server-side now — the browser never downloads the vectors.
    setDatasetSource(source ?? null)
  }

  // Unload any loaded dataset and return to the 3-card landing screen — reachable from
  // anywhere (the Header nav is always mounted), not just from within Landing itself.
  const goHome = () => {
    clear()
    setDatasetSource(null)
    setLandingView('landing')
  }

  // Same, but goes straight to the dataset picker instead of the landing cards — unless
  // no dataset has been prepared yet, in which case BYOD opens first so the user has
  // something to pick; handleBuildDone() already routes back into the picker once a
  // build finishes, so the MagniData flow continues from there.
  const goToMagniData = () => {
    clear()
    setDatasetSource(null)
    if (datasetMeta.length === 0) {
      setShowByod(true)
      return
    }
    setLandingView('datasets')
  }

  // Help is a full page (not a dialog) — reachable the same way as Home/MagniData.
  const goToHelp = () => {
    clear()
    setDatasetSource(null)
    setLandingView('help')
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
      await api.createDataset({
        name: removalName.trim(),
        description: `${kept.length} rows — ${removed.size} removed from ${data.fileName.replace(/\.csv$/i, '')}`,
        parent_source: datasetSource,
        image_names,
      })
      reloadDatasets()
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
      await api.createDataset({
        name: addName.trim(),
        description: `${picked.length} curated images from ${data.fileName.replace(/\.csv$/i, '')}`,
        parent_source: datasetSource,
        image_names,
      })
      reloadDatasets()
      setAdded(new Set()); setShowAddSave(false); setAddName('')
    } catch (e) {
      setAddMsg(`error: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setSavingAdd(false)
    }
  }, [datasetSource, data, addName, added, reloadDatasets])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', background: 'var(--c-bg)', fontFamily: 'Epilogue, sans-serif' }}>
      <Header
        data={data}
        rows={liveFilteredRows}
        filteredCount={liveFilteredRows.length}
        embeddingsAvailable={embeddingsAvailable}
        datasetSource={datasetSource}
        buildJob={buildJob.job}
        onDismissBuildJob={buildJob.dismiss}
        onGoHome={goHome}
        onOpenMagniData={goToMagniData}
        onOpenByod={() => setShowByod(true)}
        onOpenHelp={goToHelp}
      />

      {/* Main content */}
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>

        {!data && !loading && !error && tab !== 'schema' && landingView === 'landing' && (
          <Landing
            onOpenMagniData={goToMagniData}
            onOpenByod={() => setShowByod(true)}
            onOpenHelp={goToHelp}
          />
        )}

        {!data && !loading && !error && tab !== 'schema' && landingView === 'datasets' && (
          <CSVLoader onLoad={handleLoad} datasets={datasetMeta} onDelete={deleteDataset} />
        )}

        {!data && !loading && !error && tab !== 'schema' && landingView === 'help' && (
          <HelpPage onOpenByod={() => setShowByod(true)} />
        )}

        {loading && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="text-center">
              <div className="loading-spinner" />
              <p className="text-sm mt-3" style={{ color: 'var(--c-t3)' }}>Parsing CSV…</p>
            </div>
          </div>
        )}

        {error && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div className="text-center p-8">
              <p className="text-4xl mb-3">⚠</p>
              <p className="font-semibold mb-1" style={{ color: 'var(--c-t1)' }}>Failed to parse CSV</p>
              <p className="text-sm" style={{ color: 'var(--c-t3)' }}>{error}</p>
            </div>
          </div>
        )}

        {/* Tab bar (full width) + sidebar/content row below it.
            Schema is always available — doesn't need a loaded file. */}
        {(data || (!data && !loading && tab === 'schema')) && (
          <>
            {/* Tab bar — spans the full width above the sidebar */}
            <div className="tab-bar shrink-0" style={{ margin: '0', flexShrink: 0 }}>
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
              <div style={{ flex: 1 }} />

              {/* Visual-explorer plot controls — moved up here from the explorer panel.
                  Vertical split, then Color by / Sort / Fields on the right. */}
              {data && tab === 'visual-explorer' && visibleData && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginRight: 10 }}>
                  <div style={{ width: 1, height: 22, background: 'var(--c-border)', marginRight: 3 }} />
                  <span style={{ fontSize: 11, color: 'var(--c-t3)' }}>Color by</span>
                  <select value={colorBy} onChange={e => setColorBy(e.target.value)}
                          style={{ fontSize: 11, borderRadius: 7, padding: '3px 8px', outline: 'none',
                                   background: 'var(--c-surface)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }}>
                    {[...visibleData.categoricalCols, ...visibleData.numericCols].map(c => <option key={c} value={c}>{c}</option>)}
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
                <button onClick={saveCurated} disabled={!removalName.trim() || savingRemoval}
                        style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 12px', borderRadius: 6,
                                 fontSize: 11, fontWeight: 600, cursor: removalName.trim() && !savingRemoval ? 'pointer' : 'not-allowed',
                                 background: removalName.trim() && !savingRemoval ? 'var(--c-accent)' : 'var(--c-raised)',
                                 color: removalName.trim() && !savingRemoval ? '#fff' : 'var(--c-t3)',
                                 border: `1px solid ${removalName.trim() && !savingRemoval ? 'var(--c-accent)' : 'var(--c-border)'}` }}>
                  {savingRemoval ? <Loader2 size={12} style={{ animation: 'spin 0.7s linear infinite' }} /> : <Save size={12} />}
                  {savingRemoval ? 'Creating…' : 'Create dataset'}
                </button>
                <button onClick={() => setShowRemovalSave(false)} style={{ background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--c-t3)', padding: 2 }}><X size={14} /></button>
                {removalMsg && <span style={{ fontSize: 11, color: '#ef4444' }}>{removalMsg}</span>}
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
                <button onClick={saveAdded} disabled={!addName.trim() || savingAdd}
                        style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 12px', borderRadius: 6,
                                 fontSize: 11, fontWeight: 600, cursor: addName.trim() && !savingAdd ? 'pointer' : 'not-allowed',
                                 background: addName.trim() && !savingAdd ? 'var(--c-accent)' : 'var(--c-raised)',
                                 color: addName.trim() && !savingAdd ? '#fff' : 'var(--c-t3)',
                                 border: `1px solid ${addName.trim() && !savingAdd ? 'var(--c-accent)' : 'var(--c-border)'}` }}>
                  {savingAdd ? <Loader2 size={12} style={{ animation: 'spin 0.7s linear infinite' }} /> : <Save size={12} />}
                  {savingAdd ? 'Creating…' : 'Create dataset'}
                </button>
                <button onClick={() => setShowAddSave(false)} style={{ background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--c-t3)', padding: 2 }}><X size={14} /></button>
                {addMsg && <span style={{ fontSize: 11, color: '#ef4444' }}>{addMsg}</span>}
              </div>
            )}

            {/* Row below the tab bar: sidebar (left) + tab content (right) */}
            <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
              {/* Sidebar — hidden on visual-explorer and schema tabs */}
              {data && tab !== 'visual-explorer' && tab !== 'schema' && (
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
                    colorBy={colorBy} setColorBy={setColorBy}
                    sortMode={sortMode} setSortMode={setSortMode}
                    showFields={showFields} setShowFields={setShowFields}
                  />
                )}
                {data && tab === 'charts' && (
                  <Histograms data={visibleData!} rows={liveFilteredRows} />
                )}
                {data && tab === 'schema' && (
                  <Schema data={data} hiddenCols={hiddenCols} onToggleCol={toggleCol} />
                )}
              </div>
            </div>
          </>
        )}
      </div>

      {showByod && (
        <ByodDialog
          datasets={datasetMeta}
          onClose={() => setShowByod(false)}
          startUpload={buildJob.startUpload}
          startDemo={buildJob.startDemo}
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
        />
      )}
    </div>
  )
}
