import { useRef, useState } from 'react'
import { UploadCloud, Filter, AlertTriangle } from 'lucide-react'
import { api } from '../api'
import type { DatasetMeta } from '../api'
import { DatasetGraph } from './DatasetGraph'
import './CSVLoader.css'

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

      {/* Dataset lineage graph */}
      {shownDatasets.length > 0 && (
        <DatasetGraph datasets={shownDatasets} loadingSource={downloading ? shownDatasets.find(ds => ds.name === downloading)?.source ?? null : null}
                      onOpen={ds => void handleDataset(ds)} onDelete={onDelete} />
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
