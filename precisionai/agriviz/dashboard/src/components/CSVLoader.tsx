import { useRef, useState } from 'react'
import { UploadCloud, Database, ArrowRight, Loader2, Filter, Trash2, GitBranch, AlertTriangle } from 'lucide-react'
import { api } from '../api'
import type { DatasetMeta } from '../api'
import './CSVLoader.css'

// Hero gradient per card index — teal → navy → green → …
const HERO_GRADIENTS = [
  'linear-gradient(140deg, #006A7C 0%, #008eaa 100%)',
  'linear-gradient(140deg, #013755 0%, #025a8a 100%)',
  'linear-gradient(140deg, #02AA52 0%, #018f44 100%)',
]

function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`
  if (bytes >= 1_000)     return `${(bytes / 1_000).toFixed(0)} KB`
  return `${bytes} B`
}

interface Props {
  onLoad: (f: File, source?: string) => void
  datasets: DatasetMeta[]
  onDelete?: (id: number) => void
}

export function CSVLoader({ onLoad, datasets, onDelete }: Props) {
  const [dragging, setDragging]       = useState(false)
  const [downloading, setDownloading] = useState<string | null>(null)
  const [filter, setFilter]           = useState('')   // '' = show all datasets
  const [loadError, setLoadError]     = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Datasets visible after applying the dropdown filter
  const shownDatasets = filter ? datasets.filter(d => d.name === filter) : datasets

  const handleFile = (f: File | undefined) => {
    setLoadError(null)
    if (f?.name.endsWith('.csv')) onLoad(f)
  }

  const handleDataset = async (ds: DatasetMeta) => {
    if (ds.is_lfs_pointer) {
      const size = ds.lfs_size_bytes ? ` (${formatBytes(ds.lfs_size_bytes)} expected)` : ''
      setLoadError(`"${ds.name}" is only a Git LFS pointer${size}. Download the real LFS files, then reload the dashboard.`)
      return
    }
    setLoadError(null)
    setDownloading(ds.name)
    try {
      const res = await fetch(api.datasetCsvUrl(ds.source))
      if (!res.ok) {
        let message = `HTTP ${res.status}`
        try {
          const body = await res.json()
          message = body.error ?? message
          if (body.lfs_size_bytes) message += ` (${formatBytes(body.lfs_size_bytes)} expected)`
        } catch {
          // Leave the HTTP status message intact.
        }
        throw new Error(message)
      }
      const blob = await res.blob()
      onLoad(new File([blob], ds.name + '.csv', { type: 'text/csv' }), ds.source)
    } catch (e) {
      console.error('Failed to load dataset:', e)
      setLoadError(e instanceof Error ? e.message : String(e))
    } finally {
      setDownloading(null)
    }
  }

  const isLoading = downloading !== null

  return (
    <div className="loader-root">

      {/* Section header */}
      {datasets.length > 0 && (
        <div className="loader-heading">
          <h2>Select a Dataset</h2>
          <p>Pre-configured and ready to explore</p>
        </div>
      )}

      {loadError && (
        <div className="loader-alert" role="alert">
          <AlertTriangle size={15} />
          <span>{loadError}</span>
        </div>
      )}

      {/* Filter dropdown */}
      {datasets.length > 0 && (
        <div className="loader-filter">
          <Filter size={14} color="var(--c-t3)" />
          <select
            className="loader-filter-select"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            aria-label="Filter datasets"
          >
            <option value="">All datasets ({datasets.length})</option>
            {datasets.map(ds => (
              <option key={ds.source} value={ds.name}>{ds.name}</option>
            ))}
          </select>
        </div>
      )}

      {/* Card grid */}
      {shownDatasets.length > 0 && (
        <div className="dataset-grid">
          {shownDatasets.map((ds, i) => {
            const active = downloading === ds.name
            const unavailable = !!ds.is_lfs_pointer
            return (
              <div key={ds.source} style={{ position: 'relative' }}>
                {ds.deletable && ds.id != null && onDelete && (
                  <button
                    title="Delete dataset"
                    onClick={() => { if (window.confirm(`Delete dataset "${ds.name}"? This cannot be undone.`)) onDelete(ds.id!) }}
                    style={{
                      position: 'absolute', top: 8, right: 8, zIndex: 2,
                      width: 26, height: 26, borderRadius: 7, cursor: 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: 'rgba(0,0,0,0.35)', border: '1px solid rgba(255,255,255,0.25)',
                      color: '#fff', backdropFilter: 'blur(4px)',
                    }}
                  >
                    <Trash2 size={13} />
                  </button>
                )}
                <button
                  className={`ds-card${active ? ' ds-card--active' : ''}${unavailable ? ' ds-card--unavailable' : ''}`}
                  onClick={() => handleDataset(ds)}
                  disabled={isLoading}
                  aria-disabled={unavailable}
                  aria-label={`Load ${ds.name}`}
                  title={unavailable ? 'Git LFS data has not been downloaded for this dataset' : undefined}
                  style={{ width: '100%' }}
                >
                  {/* Hero */}
                  <div
                    className="ds-card-hero"
                    style={{ background: HERO_GRADIENTS[i % HERO_GRADIENTS.length] }}
                  >
                    <div className="ds-card-hero-icon">
                      {active
                        ? <Loader2 size={22} color="rgba(255,255,255,0.9)" style={{ animation: 'spin 0.7s linear infinite' }} />
                        : <Database size={22} color="rgba(255,255,255,0.9)" />
                      }
                    </div>
                    {active && <div className="ds-card-shimmer" />}
                  </div>

                  {/* Body */}
                  <div className="ds-card-body">
                    <div className="ds-card-name">{ds.name}</div>
                    <div className="ds-card-desc">{ds.description}</div>

                    {/* Badges */}
                    <div className="ds-card-meta">
                      {ds.deletable && (
                        <span className="ds-badge" style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                          <GitBranch size={10} /> derived{ds.row_count != null ? ` · ${ds.row_count.toLocaleString()} rows` : ''}
                        </span>
                      )}
                      {ds.is_lfs_pointer && (
                        <span className="ds-badge ds-badge--warn">LFS missing</span>
                      )}
                      {ds.is_lfs_pointer && ds.lfs_size_bytes != null && (
                        <span className="ds-badge">needs {formatBytes(ds.lfs_size_bytes)}</span>
                      )}
                      {!ds.is_lfs_pointer && ds.file_size_bytes != null && (
                        <span className="ds-badge">{formatBytes(ds.file_size_bytes)}</span>
                      )}
                      {ds.has_embeddings && <span className="ds-badge ds-badge--accent">embeddings</span>}
                      <span className="ds-badge ds-badge--accent">CSV</span>
                    </div>

                    {/* CTA */}
                    <div className="ds-card-cta">
                      <span className="ds-card-cta-label">
                        {active ? 'Loading…' : unavailable ? 'Download LFS first' : 'Load dataset'}
                      </span>
                      <span className="ds-card-cta-icon">
                        <ArrowRight size={13} color="var(--c-accent)" />
                      </span>
                    </div>
                  </div>
                </button>
              </div>
            )
          })}
        </div>
      )}

      {/* Divider */}
      {datasets.length > 0 && (
        <div className="loader-sep">
          <div className="loader-sep-line" />
          <span className="loader-sep-label">or upload your own CSV</span>
          <div className="loader-sep-line" />
        </div>
      )}

      {/* Upload zone */}
      <div
        className={`upload-zone${dragging ? ' upload-zone--dragging' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload a CSV file"
        onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
      >
        <div className="upload-zone-icon">
          <UploadCloud size={18} color="var(--c-accent)" />
        </div>
        <div className="upload-zone-text">
          <strong>Drop a CSV file here</strong>
          <span>or click to browse your files</span>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files?.[0])}
        />
      </div>

    </div>
  )
}
