import { useEffect, useState } from 'react'
import { X, Sparkles, UploadCloud, ArrowLeft, CheckCircle2 } from 'lucide-react'
import { api } from '../api'
import type { DatasetMeta, DemoEntry } from '../api'
import './ByodDialog.css'

type Step = 'choose' | 'demo' | 'create'

interface Props {
  datasets: DatasetMeta[]
  onClose: () => void
  startUpload: (formData: FormData) => Promise<string>
  startDemo: (demoKey: string) => Promise<string>
}

function relPath(f: File): string {
  return (f as unknown as { webkitRelativePath?: string }).webkitRelativePath || f.name
}

export function ByodDialog({ datasets, onClose, startUpload, startDemo }: Props) {
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
            {step === 'choose' ? 'Bring Your Own Data' : step === 'demo' ? 'Load Demo' : 'Create New Dataset'}
          </h3>
          <button className="byod-close" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>

        {step === 'choose' && <ChooseStep onPick={setStep} />}
        {step === 'demo' && (
          <DemoStep datasets={datasets} startDemo={startDemo} onStarted={onClose} />
        )}
        {step === 'create' && (
          <CreateStep startUpload={startUpload} onStarted={onClose} />
        )}
      </div>
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

function DemoStep({ datasets, startDemo, onStarted }: {
  datasets: DatasetMeta[]
  startDemo: (demoKey: string) => Promise<string>
  onStarted: () => void
}) {
  const [demos, setDemos] = useState<DemoEntry[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    api.listDemos().then(d => { if (alive) setDemos(d) }).catch(e => { if (alive) setError(e instanceof Error ? e.message : String(e)) })
    return () => { alive = false }
  }, [])

  const build = async (demoKey: string) => {
    setError(null)
    try {
      await startDemo(demoKey)
      onStarted()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="byod-body">
      {!demos && !error && <div className="byod-hint">Loading demos…</div>}
      {error && <div className="byod-error">{error}</div>}

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
                  <button className="byod-submit" disabled={disabled} onClick={() => build(d.key)}>
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

function CreateStep({ startUpload, onStarted }: {
  startUpload: (formData: FormData) => Promise<string>
  onStarted: () => void
}) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [images, setImages] = useState<File[]>([])
  const [annotations, setAnnotations] = useState<File[]>([])
  const [embeddings, setEmbeddings] = useState<File | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const canSubmit = !submitting && name.trim().length > 0 && images.length > 0

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
      setError(e instanceof Error ? e.message : String(e))
      setSubmitting(false)
    }
  }

  return (
    <div className="byod-body">
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
      </label>

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
      </label>

      {error && <div className="byod-error">{error}</div>}
      <button className="byod-submit byod-submit--primary" onClick={submit} disabled={!canSubmit}>
        {submitting ? 'Starting…' : 'Create Dataset'}
      </button>
    </div>
  )
}
