import { useState } from 'react'
import { GitMerge, Loader2, X } from 'lucide-react'
import { api, errorText } from '../api'
import type { DatasetMeta } from '../api'
import { canMerge } from '../lib/workspaceSelection'
import './DatasetPanel.css'

interface Props {
  datasets: DatasetMeta[]
  selection: string[]
  onMerged: (created: DatasetMeta) => void
  onClear: () => void
}

/** Side panel shown while several datasets are selected: name the union and create it. */
export function MergePanel({ datasets, selection, onMerged, onClear }: Props) {
  const [name, setName] = useState('')
  const [merging, setMerging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const chosen = selection.map(source => datasets.find(d => d.source === source)).filter((d): d is DatasetMeta => !!d)
  const rows = chosen.reduce((sum, d) => sum + (d.row_count ?? 0), 0)

  const merge = async () => {
    if (!name.trim() || !canMerge(selection) || merging) return
    setMerging(true)
    setError(null)
    try {
      onMerged(await api.mergeDatasets({ name: name.trim(), sources: selection }))
    } catch (caught) {
      setError(errorText(caught))
    } finally {
      setMerging(false)
    }
  }

  return (
    <aside className="dataset-panel" aria-label="Merge datasets">
      <header className="dataset-panel__header">
        <span className="dataset-panel__icon"><GitMerge size={19} /></span>
        <div className="dataset-panel__title">
          <h2>Merge {chosen.length} datasets</h2>
          <p>Creates a new dataset with the combined samples (duplicates by image are kept once). It starts with no experiments.</p>
        </div>
        <button className="dataset-panel__close" onClick={onClear} aria-label="Clear selection" title="Clear selection (Esc)"><X size={15} /></button>
      </header>
      <div className="dataset-panel__body">
        <ul className="dataset-panel__list">
          {chosen.map(dataset => (
            <li key={dataset.source}>
              <span><strong>{dataset.name}</strong><small>{dataset.row_count?.toLocaleString() ?? 'CSV'} samples</small></span>
            </li>
          ))}
        </ul>
        <p className="dataset-panel__foot">Up to {rows.toLocaleString()} samples before de-duplication.</p>
        <div className="dataset-panel__form">
          <label htmlFor="merge-name">New dataset name</label>
          <input id="merge-name" value={name} onChange={event => setName(event.target.value)} autoFocus
                 placeholder="Merged dataset" onKeyDown={event => { if (event.key === 'Enter') void merge() }} />
          <button className="btn-primary" onClick={() => void merge()} disabled={!name.trim() || !canMerge(selection) || merging}>
            {merging ? <Loader2 size={14} className="spin" /> : <GitMerge size={14} />} Merge datasets
          </button>
          {error && <p className="dataset-panel__note dataset-panel__error" role="alert">{error}</p>}
        </div>
      </div>
    </aside>
  )
}
