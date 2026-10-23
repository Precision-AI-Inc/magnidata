import { useState } from 'react'
import {
  ArrowRight, BarChart3, Cloud, Database, FlaskConical, GitBranch, ImageIcon, Layers, Loader2, ScanSearch, Trash2, X,
} from 'lucide-react'
import { api } from '../api'
import type { DatasetMeta, NumericColumnSummary } from '../api'
import { useDatasetDetail } from '../hooks/useDatasetDetail'
import { formatBytes, formatNumber, histogramHeights, lineageLinks, maskCoverage } from '../lib/datasetPanel'
import { formatMetric } from '../lib/experiments'
import { StatusBadge } from './experiments/ExperimentsTab'
import './DatasetPanel.css'

interface Props {
  dataset: DatasetMeta
  datasets: DatasetMeta[]
  onOpenExplorer: () => void
  onViewExperiments: () => void
  onDelete?: (id: number) => void
  onSelect: (source: string) => void
  onClose: () => void
}

const TABS = [
  ['stats', 'Statistics', BarChart3],
  ['images', 'Images', ImageIcon],
  ['annotations', 'Annotations', Layers],
  ['runs', 'Runs', FlaskConical],
  ['lineage', 'Lineage', GitBranch],
] as const
type TabKey = typeof TABS[number][0]

function Sparkline({ column }: { column: NumericColumnSummary }) {
  const heights = histogramHeights(column.histogram)
  return (
    <div className="dataset-panel__column">
      <div className="dataset-panel__column-head">
        <strong title={column.name}>{column.name}</strong>
        <small>mean {formatNumber(column.mean)}</small>
      </div>
      <svg viewBox={`0 0 ${heights.length * 8} 28`} preserveAspectRatio="none" role="img"
           aria-label={`Distribution of ${column.name}`}>
        {heights.map((h, i) => (
          <rect key={i} x={i * 8 + 1} width={6} y={28 - Math.max(h * 28, h ? 2 : 0.5)} height={Math.max(h * 28, h ? 2 : 0.5)} rx={1.5} />
        ))}
      </svg>
      <div className="dataset-panel__range"><span>{formatNumber(column.min)}</span><span>{formatNumber(column.max)}</span></div>
    </div>
  )
}

export function DatasetPanel({ dataset, datasets, onOpenExplorer, onViewExperiments, onDelete, onSelect, onClose }: Props) {
  const [tab, setTab] = useState<TabKey>('stats')
  const { summary, summaryError, runs, runsLoading } = useDatasetDetail(dataset.source)
  const links = lineageLinks(dataset, datasets)
  const coverage = summary ? maskCoverage(summary) : null
  const isCloud = dataset.name === 'Cloud Dataset'
  const samples = summary?.row_count ?? dataset.row_count

  return (
    <aside className="dataset-panel" aria-label={`${dataset.name} details`}>
      <header className="dataset-panel__header">
        <span className="dataset-panel__icon">{isCloud ? <Cloud size={19} /> : <Database size={19} />}</span>
        <div className="dataset-panel__title">
          <h2 title={dataset.name}>{dataset.name}</h2>
          <p>{dataset.description || dataset.source}</p>
        </div>
        <button className="dataset-panel__close" onClick={onClose} aria-label="Close details" title="Close (Esc)"><X size={15} /></button>
      </header>

      <div className="dataset-panel__actions">
        <button className="btn-primary" onClick={onOpenExplorer} disabled={dataset.is_lfs_pointer}
                title={dataset.is_lfs_pointer ? 'Git LFS pointer — download the real files first' : undefined}>
          <ScanSearch size={14} /> Open in Visual Explorer
        </button>
        <button className="btn-ghost" onClick={onViewExperiments}><FlaskConical size={14} /> View experiments</button>
        {dataset.deletable && dataset.id != null && onDelete && (
          <button className="dataset-panel__delete" aria-label={`Delete ${dataset.name}`} title="Delete dataset"
                  onClick={() => {
                    if (window.confirm(`Delete dataset "${dataset.name}"? This cannot be undone.`)) onDelete(dataset.id!)
                  }}>
            <Trash2 size={14} />
          </button>
        )}
      </div>

      <dl className="dataset-panel__facts">
        <div><dt>Samples</dt><dd>{samples != null ? samples.toLocaleString() : '—'}</dd></div>
        <div><dt>Columns</dt><dd>{summary ? summary.column_count : '—'}</dd></div>
        <div><dt>File size</dt><dd>{formatBytes(dataset.file_size_bytes)}</dd></div>
        <div><dt>Runs</dt><dd>{runsLoading ? '…' : runs.length >= 8 ? '8+' : runs.length}</dd></div>
      </dl>

      <div className="dataset-panel__tabs" role="tablist">
        {TABS.map(([key, text, Icon]) => (
          <button key={key} role="tab" aria-selected={tab === key} data-active={tab === key} onClick={() => setTab(key)}>
            <Icon size={13} /> {text}
          </button>
        ))}
      </div>

      <div className="dataset-panel__body" role="tabpanel" key={tab}>
        {dataset.is_lfs_pointer && <p className="dataset-panel__note">This CSV is only a Git LFS pointer. Run <code>git lfs pull</code> to see its statistics.</p>}
        {!summary && !summaryError && !dataset.is_lfs_pointer && (tab === 'stats' || tab === 'images' || tab === 'annotations') && (
          <div className="dataset-panel__loading"><Loader2 size={15} className="spin" /> Reading dataset…</div>
        )}
        {summaryError && (tab === 'stats' || tab === 'images' || tab === 'annotations') && (
          <p className="dataset-panel__note" role="alert">{summaryError}</p>
        )}

        {tab === 'stats' && summary && (
          <>
            {summary.numeric_columns.length === 0
              ? <p className="dataset-panel__note">No numeric feature columns in this dataset.</p>
              : <div className="dataset-panel__columns">{summary.numeric_columns.map(c => <Sparkline key={c.name} column={c} />)}</div>}
            <p className="dataset-panel__foot">
              {summary.numeric_column_count} numeric of {summary.column_count} columns
              {summary.truncated && ' · distributions use the first 20,000 rows'}
            </p>
          </>
        )}

        {tab === 'images' && summary && (
          summary.sample_images.length
            ? (
              <div className="dataset-panel__images">
                {summary.sample_images.map(path => (
                  <a key={path} href={api.imageUrl(path, 1600)} target="_blank" rel="noreferrer" title={path.split('/').pop()}>
                    <img src={api.imageUrl(path, 260)} alt={path.split('/').pop() ?? 'Sample image'} loading="lazy" />
                  </a>
                ))}
              </div>
            )
            : <p className="dataset-panel__note">This dataset has no image column.</p>
        )}

        {tab === 'annotations' && summary && coverage && (
          <div className="dataset-panel__annotations">
            <div>
              <div className="dataset-panel__meter" role="meter" aria-valuemin={0} aria-valuemax={100}
                   aria-valuenow={Math.round((coverage.fraction ?? 0) * 100)}>
                <span style={{ width: `${(coverage.fraction ?? 0) * 100}%` }} />
              </div>
              <p className="dataset-panel__foot">{coverage.label}</p>
            </div>
            <ul className="dataset-panel__list">
              <li><span>Cluster partitions committed</span><strong>{summary.partitions_committed.toLocaleString()}</strong></li>
              <li><span>Embeddings</span><strong>{summary.has_embeddings ? 'Available' : 'None'}</strong></li>
            </ul>
          </div>
        )}

        {tab === 'runs' && (
          runsLoading
            ? <div className="dataset-panel__loading"><Loader2 size={15} className="spin" /> Loading runs…</div>
            : runs.length === 0
              ? <p className="dataset-panel__note">No training runs on this dataset yet. Open it in the Visual Explorer to launch one.</p>
              : (
                <ul className="dataset-panel__list">
                  {runs.map(run => (
                    <li key={run.run_id}>
                      <span>
                        <strong>{run.name || `Run ${run.run_id.slice(0, 8)}`}</strong>
                        <small>val mIoU {formatMetric(run.metrics['val_metric/miou'])}</small>
                      </span>
                      <StatusBadge status={run.status} />
                    </li>
                  ))}
                </ul>
              )
        )}

        {tab === 'lineage' && (
          <div className="dataset-panel__lineage">
            {([['Derived from', links.parents], ['Derived datasets', links.children]] as const).map(([title, items]) => (
              <section key={title}>
                <h3>{title}</h3>
                {items.length === 0
                  ? <p className="dataset-panel__note">{title === 'Derived from' ? 'A root dataset.' : 'Nothing derived from it yet.'}</p>
                  : items.map(item => (
                    <button key={item.source} onClick={() => onSelect(item.source)}>
                      <Database size={13} /> <span>{item.name}</span> <ArrowRight size={12} />
                    </button>
                  ))}
              </section>
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}
