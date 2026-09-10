import { useEffect, useState } from 'react'
import { X, Sparkles, UploadCloud, ArrowLeft, CheckCircle2, FolderOpen, AlertTriangle, Loader2 } from 'lucide-react'
import { api, errorText } from '../api'
import type { BuildJobStatus, BuildLimits, DatasetMeta, DemoEntry, LocalFolder } from '../api'
import './ByodDialog.css'

type Step = 'choose' | 'demo' | 'local' | 'create'

interface Props {
  datasets: DatasetMeta[]
  buildJob?: BuildJobStatus | null
  buildBusy?: boolean
  onClose: () => void
  startUpload: (formData: FormData) => Promise<string>
  startDemo: (demoKey: string) => Promise<string>
  startLocal: (folder: string, name?: string) => Promise<string>
}

function relPath(f: File): string {
  return (f as unknown as { webkitRelativePath?: string }).webkitRelativePath || f.name
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${bytes} B`
}

export function ByodDialog({ datasets, buildJob, buildBusy = false, onClose, startUpload, startDemo, startLocal }: Props) {
  const [step, setStep] = useState<Step>('choose')

  return (
    <div className="byod-overlay" role="dialog" aria-modal="true">
      <div className="byod-panel">
        <div className="byod-header">
          {step !== 'choose' && (
            <button className="byod-back" onClick={() => setStep('choose')} aria-label="Back">
              <ArrowLeft size={15} />
            </button>
          )}
          <h3>
            {step === 'choose'
              ? 'Bring Your Own Data'
              : step === 'demo'
                ? 'Load Demo'
                : step === 'local'
                  ? 'Prepare From Server Folder'
                  : 'Create New Dataset'}
          </h3>
          <button className="byod-close" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>

        {/* CreateStep shows its own notice — it must stay quiet while its own upload runs. */}
        {buildBusy && step !== 'create' && (
          <div className="byod-notice-row"><BusyNotice job={buildJob} /></div>
        )}

        {step === 'choose' && <ChooseStep onPick={setStep} />}
        {step === 'demo' && (
          <DemoStep datasets={datasets} busy={buildBusy} startDemo={startDemo} onStarted={onClose} />
        )}
        {step === 'local' && <LocalStep busy={buildBusy} startLocal={startLocal} onStarted={onClose} />}
        {step === 'create' && (
          <CreateStep buildJob={buildJob} busy={buildBusy} startUpload={startUpload} onStarted={onClose} />
        )}
      </div>
    </div>
  )
}

function ErrorNotice({ message }: { message: string }) {
  return (
    <div className="byod-error" role="alert">
      <AlertTriangle size={15} />
      <span>{message}</span>
    </div>
  )
}

/** Explains why every Prepare/Create button is disabled: the API runs one build at a time. */
function BusyNotice({ job }: { job?: BuildJobStatus | null }) {
  const percent = job ? Math.max(0, Math.min(100, Math.round(job.percent ?? 0))) : null
  return (
    <div className="byod-busy" role="status">
      <Loader2 size={15} className="byod-spin" />
      <span>
        {job?.name ? <strong>{job.name}</strong> : 'A dataset'} is still being prepared
        {percent !== null ? ` (${percent}%)` : ''}. MagniData prepares one dataset at a time —
        you can start another once it finishes.
      </span>
    </div>
  )
}

function ChooseStep({ onPick }: { onPick: (s: Step) => void }) {
  return (
    <div className="byod-body">
      <button className="byod-choice" onClick={() => onPick('demo')}>
        <Sparkles size={18} />
        <div>
          <div className="byod-choice-title">Load Demo</div>
          <div className="byod-choice-desc">Prepare a ready-made sample dataset in one click</div>
        </div>
      </button>
      <button className="byod-choice" onClick={() => onPick('local')}>
        <FolderOpen size={18} />
        <div>
          <div className="byod-choice-title">Prepare From Server Folder</div>
          <div className="byod-choice-desc">Build from a folder already placed in image_sets/ — no upload, no size limit</div>
        </div>
      </button>
      <button className="byod-choice" onClick={() => onPick('create')}>
        <UploadCloud size={18} />
        <div>
          <div className="byod-choice-title">Create New Dataset</div>
          <div className="byod-choice-desc">Upload your own images, and optionally annotations and pre-computed embeddings</div>
        </div>
      </button>
    </div>
  )
}

function DemoStep({ datasets, busy, startDemo, onStarted }: {
  datasets: DatasetMeta[]
  busy: boolean
  startDemo: (demoKey: string) => Promise<string>
  onStarted: () => void
}) {
  const [demos, setDemos] = useState<DemoEntry[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    api.listDemos().then(d => { if (alive) setDemos(d) }).catch(e => { if (alive) setError(errorText(e)) })
    return () => { alive = false }
  }, [])

  const build = async (demoKey: string) => {
    setError(null)
    try {
      await startDemo(demoKey)
      onStarted()
    } catch (e) {
      setError(errorText(e))
    }
  }

  return (
    <div className="byod-body">
      {!demos && !error && <div className="byod-hint">Loading demos…</div>}
      {error && <ErrorNotice message={error} />}

      <div className="byod-demo-list">
        {demos?.map(d => {
          // Matched by the backend's registration path convention (data_user/<demo_key>.csv),
          // not by display name — confi.yaml can independently have a dataset with the same
          // name (e.g. a curated "COCO128" entry), which must not be mistaken for "already built".
          const alreadyBuilt = datasets.some(ds => ds.source === `data_user/${d.key}.csv`)
          const disabled = !d.enabled || alreadyBuilt
          return (
            <div key={d.key} className={`byod-demo-row${disabled ? ' byod-demo-row--disabled' : ''}`}>
              <div>
                <div className="byod-demo-name">{d.name}</div>
                <div className="byod-demo-desc">
                  {alreadyBuilt ? 'Already added to your datasets' : d.description}
                </div>
              </div>
              {alreadyBuilt
                ? <span className="byod-demo-badge"><CheckCircle2 size={13} /> Added</span>
                : (
                  <button className="byod-submit" disabled={disabled || busy} onClick={() => build(d.key)}>
                    {d.enabled ? 'Prepare' : 'Coming soon'}
                  </button>
                )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function LocalStep({ busy, startLocal, onStarted }: {
  busy: boolean
  startLocal: (folder: string, name?: string) => Promise<string>
  onStarted: () => void
}) {
  const [folders, setFolders] = useState<LocalFolder[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    api.listLocalFolders()
      .then(f => { if (alive) setFolders(f) })
      .catch(e => { if (alive) setError(errorText(e)) })
    return () => { alive = false }
  }, [])

  const build = async (folder: string) => {
    setError(null)
    setStarting(folder)
    try {
      await startLocal(folder)
      onStarted()
    } catch (e) {
      setError(errorText(e))
      setStarting(null)
    }
  }

  return (
    <div className="byod-body">
      <div className="byod-hint">
        Folders placed in <code>image_sets/</code> on the server, each holding an{' '}
        <code>images/</code> directory and optionally a matching <code>labels/</code> directory.
        Images are read where they are — nothing is copied or uploaded.
      </div>
      {!folders && !error && <div className="byod-hint">Indexing image_sets/…</div>}
      {error && <ErrorNotice message={error} />}
      {folders?.length === 0 && (
        <div className="byod-hint">
          No image sets found. Copy a folder into <code>image_sets/</code> and reopen this dialog.
        </div>
      )}

      <div className="byod-demo-list">
        {folders?.map(f => (
          <div key={f.folder} className={`byod-demo-row${f.built ? ' byod-demo-row--disabled' : ''}`}>
            <div>
              <div className="byod-demo-name">{f.folder}</div>
              <div className="byod-demo-desc">
                {f.built
                  ? 'Already added to your datasets'
                  : `${f.image_count.toLocaleString()} images · ${f.label_count.toLocaleString()} labels`}
              </div>
            </div>
            {f.built
              ? <span className="byod-demo-badge"><CheckCircle2 size={13} /> Added</span>
              : (
                <button className="byod-submit" disabled={starting !== null || busy} onClick={() => build(f.folder)}>
                  {starting === f.folder ? 'Starting…' : 'Prepare'}
                </button>
              )}
          </div>
        ))}
      </div>
    </div>
  )
}

function CreateStep({ buildJob, busy, startUpload, onStarted }: {
  buildJob?: BuildJobStatus | null
  busy: boolean
  startUpload: (formData: FormData) => Promise<string>
  onStarted: () => void
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [images, setImages] = useState<File[]>([])
  const [annotations, setAnnotations] = useState<File[]>([])
  const [embeddings, setEmbeddings] = useState<File | null>(null)
  const [limits, setLimits] = useState<BuildLimits | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    api.buildLimits().then(l => { if (alive) setLimits(l) }).catch(() => {})
    return () => { alive = false }
  }, [])

  const imageBytes = images.reduce((sum, f) => sum + f.size, 0)
  const annotationBytes = annotations.reduce((sum, f) => sum + f.size, 0)
  const uploadBytes = imageBytes + annotationBytes
  const embeddingsBytes = embeddings?.size ?? 0
  const limitError = limits && (
    images.length > limits.max_images ||
    uploadBytes > limits.max_upload_bytes ||
    embeddingsBytes > limits.max_embeddings_bytes
  )
  const canSubmit = !submitting && !busy && name.trim().length > 0 && images.length > 0 && !limitError

  const submit = async () => {
    setSubmitting(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('source_type', 'upload')
      formData.append('name', name.trim())
      formData.append('description', description.trim())
      for (const f of images) formData.append('images', f, relPath(f))
      for (const f of annotations) formData.append('annotations', f, relPath(f))
      if (embeddings) formData.append('embeddings', embeddings, embeddings.name)
      await startUpload(formData)
      onStarted()
    } catch (e) {
      setError(errorText(e))
      setSubmitting(false)
    }
  }

  return (
    <div className="byod-body">
      {busy && !submitting && <BusyNotice job={buildJob} />}
      <label className="byod-field">
        <span>Name</span>
        <input value={name} onChange={e => setName(e.target.value)} placeholder="My Field Survey" />
      </label>
      <label className="byod-field">
        <span>Description (optional)</span>
        <input value={description} onChange={e => setDescription(e.target.value)} />
      </label>

      <label className="byod-field">
        <span>Images</span>
        <div className="byod-dropzone" onClick={() => document.getElementById('byod-images-input')?.click()}>
          <UploadCloud size={18} />
          <span>{images.length > 0 ? `${images.length} image(s) selected` : 'Choose a folder or files'}</span>
        </div>
        <input
          id="byod-images-input" type="file" multiple accept="image/*"
          // @ts-expect-error -- webkitdirectory is a non-standard attribute
          webkitdirectory=""
          style={{ display: 'none' }}
          onChange={e => setImages(Array.from(e.target.files || []))}
        />
        <SelectionSize
          count={images.length}
          bytes={imageBytes}
          maxCount={limits?.max_images}
          overCount={!!limits && images.length > limits.max_images}
        />
      </label>

      <label className="byod-field">
        <span>Annotations (optional — COCO-format JSON, matched to images by filename)</span>
        <div className="byod-dropzone" onClick={() => document.getElementById('byod-annotations-input')?.click()}>
          <UploadCloud size={18} />
          <span>{annotations.length > 0 ? `${annotations.length} annotation file(s) selected` : 'Choose annotation files'}</span>
        </div>
        <input
          id="byod-annotations-input" type="file" multiple accept="application/json"
          // @ts-expect-error -- webkitdirectory is a non-standard attribute
          webkitdirectory=""
          style={{ display: 'none' }}
          onChange={e => setAnnotations(Array.from(e.target.files || []))}
        />
        {annotations.length > 0 && (
          <SelectionSize
            count={annotations.length}
            bytes={annotationBytes}
          />
        )}
      </label>

      {(images.length > 0 || annotations.length > 0) && (
        <PayloadSize
          bytes={uploadBytes}
          maxBytes={limits?.max_upload_bytes}
          overBytes={!!limits && uploadBytes > limits.max_upload_bytes}
        />
      )}

      <label className="byod-field">
        <span>Embeddings (optional — a pre-computed embeddings JSON; MagniData doesn't compute embeddings itself)</span>
        <div className="byod-dropzone" onClick={() => document.getElementById('byod-embeddings-input')?.click()}>
          <UploadCloud size={18} />
          <span>{embeddings ? embeddings.name : 'Choose an embeddings.json file'}</span>
        </div>
        <input
          id="byod-embeddings-input" type="file" accept="application/json"
          style={{ display: 'none' }}
          onChange={e => setEmbeddings(e.target.files?.[0] ?? null)}
        />
        {embeddings && (
          <SelectionSize
            count={1}
            bytes={embeddingsBytes}
            maxBytes={limits?.max_embeddings_bytes}
            overBytes={!!limits && embeddingsBytes > limits.max_embeddings_bytes}
          />
        )}
      </label>

      {submitting && <BuildProgress job={buildJob} />}
      {error && <ErrorNotice message={error} />}
      <button className="byod-submit byod-submit--primary" onClick={submit} disabled={!canSubmit}>
        {submitting ? (buildJob?.status === 'uploading' ? 'Uploading…' : 'Starting…') : 'Create Dataset'}
      </button>
    </div>
  )
}

function SelectionSize({ count, bytes, maxCount, maxBytes, overCount, overBytes }: {
  count: number
  bytes: number
  maxCount?: number
  maxBytes?: number
  overCount?: boolean
  overBytes?: boolean
}) {
  const status = [
    `${count.toLocaleString()} selected`,
    maxCount ? `${maxCount.toLocaleString()} max` : null,
    `${formatBytes(bytes)}`,
    maxBytes ? `${formatBytes(maxBytes)} max` : null,
  ].filter(Boolean).join(' · ')

  return (
    <div className={`byod-size${overCount || overBytes ? ' byod-size--error' : ''}`}>
      {status}
    </div>
  )
}

function PayloadSize({ bytes, maxBytes, overBytes }: {
  bytes: number
  maxBytes?: number
  overBytes?: boolean
}) {
  return (
    <div className={`byod-size${overBytes ? ' byod-size--error' : ''}`}>
      Images + annotations: {formatBytes(bytes)}{maxBytes ? ` / ${formatBytes(maxBytes)} max` : ''}
    </div>
  )
}

function BuildProgress({ job }: { job?: BuildJobStatus | null }) {
  if (!job || job.status === 'done' || job.status === 'error') return null
  const percent = Math.max(0, Math.min(100, Math.round(job.percent ?? 0)))

  return (
    <div className="byod-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
      <div className="byod-progress-row">
        <span>{job.message}</span>
        <span>{percent}%</span>
      </div>
      <div className="byod-progress-track">
        <div className="byod-progress-fill" style={{ width: `${percent}%` }} />
      </div>
    </div>
  )
}
