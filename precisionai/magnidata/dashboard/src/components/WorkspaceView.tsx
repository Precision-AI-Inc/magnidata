import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Activity, CheckCircle2, Database, FlaskConical, Loader2, Plus, Sparkles, UploadCloud } from 'lucide-react'
import { api } from '../api'
import type { DatasetMeta, MlflowStatus, RunSummary, TrainingPresets } from '../api'
import { formatBytes } from '../lib/datasetPanel'
import { nextSelection, readWorkspaceSelection, rememberWorkspaceSelection } from '../lib/workspaceSelection'
import { useExperimentCounts } from '../hooks/useExperimentCounts'
import { DatasetGraph } from './DatasetGraph'
import { DatasetPanel } from './DatasetPanel'
import { MergePanel } from './MergePanel'
import './WorkspaceView.css'

interface Props {
  datasets: DatasetMeta[]
  onOpenDataset: (source: string) => void
  onViewExperiments: (source: string) => void
  onDelete: (id: number) => void
  onOpenByod: () => void
  /** Why adding datasets and viewing experiments are off on this server, shown as their tooltip. */
  demoReason?: string
  onUploadFile: (file: File) => void
  /** Reload the dataset catalogue after a merge created a new dataset. */
  onDatasetsChanged: () => Promise<unknown> | void
}

interface ServiceState {
  runs: RunSummary[]
  mlflow: MlflowStatus | null
  presets: TrainingPresets | null
  loading: boolean
}

function useServices(): ServiceState {
  const [state, setState] = useState<ServiceState>({ runs: [], mlflow: null, presets: null, loading: true })
  useEffect(() => {
    let cancelled = false
    Promise.allSettled([api.getMlflowStatus(), api.listExperiments(), api.getTrainingPresets()]).then(async results => {
      const mlflow = results[0].status === 'fulfilled' ? results[0].value : null
      const experiments = results[1].status === 'fulfilled' ? results[1].value.experiments : []
      const presets = results[2].status === 'fulfilled' ? results[2].value : null
      let runs: RunSummary[] = []
      if (experiments.length) {
        try {
          runs = (await api.searchRuns({
            experimentIds: experiments.map(e => e.experiment_id),
            orderBy: 'attributes.start_time DESC',
            maxResults: 50,
          })).runs
        } catch { /* the trainer card still explains an unavailable MLflow service */ }
      }
      if (!cancelled) setState({ runs, mlflow, presets, loading: false })
    })
    return () => { cancelled = true }
  }, [])
  return state
}

export function WorkspaceView({ datasets, onOpenDataset, onViewExperiments, onDelete, onOpenByod, demoReason, onUploadFile, onDatasetsChanged }: Props) {
  const services = useServices()
  const upload = useRef<HTMLInputElement>(null)
  const experiments = useExperimentCounts()
  const [selection, setSelectionState] = useState<string[]>(() => {
    const remembered = readWorkspaceSelection()
    return remembered ? [remembered] : []
  })

  const setSelection = useCallback((next: string[]) => {
    setSelectionState(next)
    rememberWorkspaceSelection(next.length === 1 ? next[0] : null)
  }, [])
  const select = useCallback((source: string | null, additive: boolean) => {
    setSelection(source === null ? [] : nextSelection(selection, source, additive))
  }, [selection, setSelection])

  const known = useMemo(() => selection.filter(source => datasets.some(d => d.source === source)), [datasets, selection])
  const dataset = useMemo(
    () => (known.length === 1 ? datasets.find(d => d.source === known[0]) ?? null : null),
    [datasets, known],
  )
  const merging = known.length > 1
  const panelOpen = dataset !== null || merging

  const totals = useMemo(() => ({
    samples: datasets.reduce((sum, d) => sum + (d.row_count ?? 0), 0),
    bytes: datasets.reduce((sum, d) => sum + (d.file_size_bytes ?? 0), 0),
    explorerReady: datasets.filter(d => d.has_embeddings).length,
  }), [datasets])

  const active = services.runs.filter(run => run.status === 'RUNNING' || run.status === 'SCHEDULED').length
  const trainerReady = !!services.mlflow?.reachable && !!services.presets?.backend.available
  const trainerDetail = services.loading
    ? 'checking services'
    : trainerReady
      ? (active ? `${active} run${active === 1 ? '' : 's'} in progress` : 'worker idle')
      : (services.presets?.backend.reason || (services.mlflow?.reachable ? 'no worker heartbeat' : 'MLflow unavailable'))

  return (
    <div className="workspace-view">
      <section className="workspace-summary" aria-label="Workspace summary">
        <SummaryCard icon={Database} tone="teal" label="Datasets" value={String(datasets.length)} detail={`${formatBytes(totals.bytes)} of CSV on disk`} />
        <SummaryCard icon={Activity} tone="blue" label="Samples" value={totals.samples.toLocaleString()} detail="across all datasets" />
        <SummaryCard icon={Sparkles} tone="green" label="Explorer ready" value={String(totals.explorerReady)} detail="with embeddings" />
        <SummaryCard icon={FlaskConical} tone="amber" label="Runs" value={services.loading ? '…' : String(services.runs.length)} detail={active ? `${active} active now` : 'recent training runs'} active={active > 0} />
        <SummaryCard icon={trainerReady ? CheckCircle2 : Loader2} tone={trainerReady ? 'green' : 'amber'} label="Trainer" value={services.loading ? '…' : trainerReady ? 'Ready' : 'Offline'} detail={trainerDetail} />
      </section>

      <div className="workspace-toolbar">
        <div>
          <h2>Dataset lineage</h2>
          <p>Select a dataset to inspect it, then open it in the Visual Explorer or review its experiments.</p>
        </div>
        <div className="workspace-toolbar__actions">
          <button className="btn-ghost" onClick={() => upload.current?.click()} disabled={!!demoReason} title={demoReason}
                  style={demoReason ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}><UploadCloud size={14} /> Upload CSV</button>
          <button className="btn-primary" onClick={onOpenByod} disabled={!!demoReason} title={demoReason}
                  style={demoReason ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}><Plus size={14} /> Add dataset</button>
          <input ref={upload} type="file" accept=".csv" hidden
                 onChange={event => {
                   const file = event.target.files?.[0]
                   if (file?.name.endsWith('.csv')) onUploadFile(file)
                   event.target.value = ''
                 }} />
        </div>
      </div>

      <div className="workspace-stage" data-panel={panelOpen ? 'open' : 'closed'}>
        <div className="workspace-stage__graph">
          {datasets.length > 0
            ? <DatasetGraph datasets={datasets} selection={known} experiments={experiments} onSelect={select}
                            onOpen={onOpenDataset} onDelete={id => { setSelection([]); onDelete(id) }} />
            : (
              <button className="workspace-empty" onClick={onOpenByod} disabled={!!demoReason} title={demoReason}
                      style={demoReason ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}>
                <UploadCloud size={22} />
                <span><strong>Add your first dataset</strong><small>Upload images and labels or load a demo.</small></span>
              </button>
            )}
        </div>
        {panelOpen && (
          <div className="workspace-stage__panel">
            {dataset ? (
              <DatasetPanel
                dataset={dataset}
                datasets={datasets}
                onOpenExplorer={() => onOpenDataset(dataset.source)}
                onViewExperiments={() => onViewExperiments(dataset.source)}
                experimentsDisabled={demoReason}
                onDelete={id => { setSelection([]); onDelete(id) }}
                onSelect={source => setSelection([source])}
                onClose={() => setSelection([])}
              />
            ) : (
              <MergePanel
                datasets={datasets}
                selection={known}
                onClear={() => setSelection([])}
                onMerged={async created => {
                  await onDatasetsChanged()
                  setSelection([created.source])
                }}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function SummaryCard({ icon: Icon, tone, label, value, detail, active = false }: {
  icon: typeof Database
  tone: 'teal' | 'green' | 'amber' | 'blue'
  label: string
  value: string
  detail: string
  active?: boolean
}) {
  return (
    <article className="workspace-summary-card surface">
      <span className={`workspace-summary-card__icon workspace-summary-card__icon--${tone}`}>
        <Icon size={21} className={active ? 'workspace-summary-card__spinner' : undefined} />
      </span>
      <span className="workspace-summary-card__copy"><small>{label}</small><strong>{value}</strong></span>
      <span className="workspace-summary-card__detail" title={detail}>{detail}</span>
    </article>
  )
}
