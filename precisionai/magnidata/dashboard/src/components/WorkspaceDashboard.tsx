import { useEffect, useState } from 'react'
import {
  ArrowRight,
  CheckCircle2,
  Database,
  FlaskConical,
  HelpCircle,
  ImageIcon,
  Loader2,
  ScanSearch,
  Server,
  UploadCloud,
} from 'lucide-react'
import { api } from '../api'
import type { DatasetMeta, MlflowStatus, RunSummary, TrainingPresets } from '../api'
import { StatusBadge } from './experiments/ExperimentsTab'

interface Props {
  datasets: DatasetMeta[]
  onOpenDataset: (source: string) => void
  onOpenDatasets: () => void
  onOpenExplorer: () => void
  onOpenByod: () => void
  onOpenExperiments: () => void
  onOpenHelp: () => void
}

interface WorkspaceState {
  runs: RunSummary[]
  mlflow: MlflowStatus | null
  presets: TrainingPresets | null
}

function relativeTime(timestamp: number | null | undefined): string {
  if (!timestamp) return '—'
  const seconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000))
  if (seconds < 60) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}

function datasetSize(dataset: DatasetMeta): string {
  if (dataset.row_count != null) return `${dataset.row_count.toLocaleString()} samples`
  if (dataset.file_size_bytes != null) return `${(dataset.file_size_bytes / 1_000_000).toFixed(1)} MB`
  return dataset.has_embeddings ? 'Embeddings ready' : 'Ready to explore'
}

export function WorkspaceDashboard({
  datasets,
  onOpenDataset,
  onOpenDatasets,
  onOpenExplorer,
  onOpenByod,
  onOpenExperiments,
  onOpenHelp,
}: Props) {
  const [state, setState] = useState<WorkspaceState>({ runs: [], mlflow: null, presets: null })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    Promise.allSettled([api.getMlflowStatus(), api.listExperiments(), api.getTrainingPresets()]).then(async results => {
      if (cancelled) return
      const mlflow = results[0].status === 'fulfilled' ? results[0].value : null
      const experiments = results[1].status === 'fulfilled' ? results[1].value.experiments : []
      const presets = results[2].status === 'fulfilled' ? results[2].value : null
      let runs: RunSummary[] = []
      if (experiments.length) {
        try {
          runs = (await api.searchRuns({
            experimentIds: experiments.map(e => e.experiment_id),
            orderBy: 'attributes.start_time DESC',
            maxResults: 6,
          })).runs
        } catch { /* The status card still explains an unavailable MLflow service. */ }
      }
      if (!cancelled) {
        setState({ runs, mlflow, presets })
        setLoading(false)
      }
    })
    return () => { cancelled = true }
  }, [])

  const recentDatasets = datasets.slice(0, 4)
  const running = state.runs.filter(run => run.status === 'RUNNING' || run.status === 'SCHEDULED').length
  const completed = state.runs.filter(run => run.status === 'FINISHED').length

  return (
    <div className="workspace-dashboard">
      <section className="workspace-quick surface" aria-labelledby="quick-actions-title">
        <div className="workspace-section-heading">
          <div>
            <h2 id="quick-actions-title">Quick actions</h2>
            <p>Move from raw images to a measurable training run.</p>
          </div>
        </div>
        <div className="workspace-quick__grid">
          <QuickAction icon={UploadCloud} label="Upload dataset" detail="Images, labels, or CSV" onClick={onOpenByod} />
          <QuickAction icon={ScanSearch} label="Visual explorer" detail="Embeddings and feature axes" onClick={onOpenExplorer} />
          <QuickAction icon={FlaskConical} label="Experiments" detail="Launch and compare runs" onClick={onOpenExperiments} />
          <QuickAction icon={HelpCircle} label="Setup guide" detail="Data and training contracts" onClick={onOpenHelp} />
        </div>
      </section>

      <div className="workspace-grid">
        <section className="workspace-panel surface">
          <div className="workspace-section-heading">
            <div>
              <h2>Recent datasets</h2>
              <p>{datasets.length} available in this workspace</p>
            </div>
            <button onClick={onOpenDatasets}>View all <ArrowRight size={13} /></button>
          </div>
          <div className="workspace-list">
            {recentDatasets.map((dataset, index) => (
              <button className="workspace-list__row" key={dataset.source} onClick={() => onOpenDataset(dataset.source)}>
                <span className={`workspace-dataset-icon workspace-dataset-icon--${index % 3}`}>
                  {dataset.has_embeddings ? <ImageIcon size={17} /> : <Database size={17} />}
                </span>
                <span className="workspace-list__main">
                  <strong>{dataset.name}</strong>
                  <small>{datasetSize(dataset)}{dataset.has_embeddings ? ' · embeddings' : ''}</small>
                </span>
                <ArrowRight size={14} />
              </button>
            ))}
            {!recentDatasets.length && (
              <button className="workspace-empty" onClick={onOpenByod}>
                <UploadCloud size={20} />
                <span><strong>Add your first dataset</strong><small>Upload images and labels or load a demo.</small></span>
              </button>
            )}
          </div>
        </section>

        <section className="workspace-panel surface">
          <div className="workspace-section-heading">
            <div>
              <h2>Recent experiments</h2>
              <p>{running} active · {completed} completed in this view</p>
            </div>
            <button onClick={onOpenExperiments}>View all <ArrowRight size={13} /></button>
          </div>
          <div className="workspace-list">
            {loading && <div className="workspace-loading"><Loader2 size={16} /> Loading runs…</div>}
            {!loading && state.runs.slice(0, 4).map(run => (
              <button className="workspace-list__row" key={run.run_id} onClick={onOpenExperiments}>
                <span className="workspace-run-icon"><FlaskConical size={16} /></span>
                <span className="workspace-list__main">
                  <strong>{run.name || `Run ${run.run_id.slice(0, 8)}`}</strong>
                  <small>{run.magnidata['magnidata.dataset_source']?.split('/').pop() || 'Unlinked dataset'} · {relativeTime(run.start_time)}</small>
                </span>
                <StatusBadge status={run.status} />
              </button>
            ))}
            {!loading && !state.runs.length && (
              <div className="workspace-empty workspace-empty--static">
                <FlaskConical size={20} />
                <span><strong>No experiment runs yet</strong><small>Open a dataset to launch your first training run.</small></span>
              </div>
            )}
          </div>
        </section>
      </div>

      <section className="workspace-health surface">
        <div className="workspace-section-heading">
          <div>
            <h2>Workspace status</h2>
            <p>Live service readiness for the training feedback loop.</p>
          </div>
        </div>
        <div className="workspace-health__grid">
          <HealthItem icon={Database} label="Datasets" value={`${datasets.length} registered`} ok />
          <HealthItem icon={Server} label="MLflow" value={state.mlflow?.reachable ? 'Connected' : 'Unavailable'} ok={!!state.mlflow?.reachable} />
          <HealthItem icon={FlaskConical} label="Trainer" value={state.presets?.backend.available ? 'Worker ready' : (state.presets?.backend.reason || 'No heartbeat')} ok={!!state.presets?.backend.available} />
          <HealthItem icon={CheckCircle2} label="Feedback loop" value={datasets.some(d => d.has_embeddings) ? 'Ready to explore' : 'Dataset required'} ok={datasets.some(d => d.has_embeddings)} />
        </div>
      </section>
    </div>
  )
}

function QuickAction({ icon: Icon, label, detail, onClick }: { icon: typeof UploadCloud; label: string; detail: string; onClick: () => void }) {
  return (
    <button className="workspace-action" onClick={onClick}>
      <span><Icon size={22} /></span>
      <strong>{label}</strong>
      <small>{detail}</small>
    </button>
  )
}

function HealthItem({ icon: Icon, label, value, ok }: { icon: typeof Database; label: string; value: string; ok: boolean }) {
  return (
    <div className="workspace-health__item">
      <span className="workspace-health__icon"><Icon size={17} /></span>
      <span><small>{label}</small><strong>{value}</strong></span>
      <span className="workspace-health__indicator" data-ok={ok} />
    </div>
  )
}
