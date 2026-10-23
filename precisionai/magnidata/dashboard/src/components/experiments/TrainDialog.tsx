import { useMemo, useState } from 'react'
import { Loader2, X } from 'lucide-react'
import { api, errorText } from '../../api'
import type { TrainingPresets } from '../../api'
import { launchParams, validateParams } from '../../lib/train'
import type { FieldErrors, TrainParams } from '../../lib/train'
import '../ByodDialog.css'

interface Props {
  presets: TrainingPresets
  datasetSource: string
  datasetName: string
  parentRunId?: string
  /** Settings to start from instead of the preset's defaults (e.g. the run being retrained). */
  initialParams?: Partial<TrainParams>
  /** Preset to preselect (the retrained run's). */
  initialPreset?: string
  onClose: () => void
  onLaunched: (runId: string) => void
}

const FIELDS: { key: keyof TrainParams; label: string; step?: number }[] = [
  { key: 'epochs', label: 'Epochs' },
  { key: 'score_every', label: 'Score every N epochs' },
  { key: 'batch_size', label: 'Batch size' },
  { key: 'image_size', label: 'Image size', step: 32 },
  { key: 'lr', label: 'Learning rate', step: 0.0001 },
  { key: 'val_fraction', label: 'Validation fraction', step: 0.05 },
  { key: 'top_k_predictions', label: 'Hardest predictions to save' },
  { key: 'seed', label: 'Seed' },
]

/** AgriTune optimisation settings, shown with its other controls. */
const AGRITUNE_NUMBERS: { key: keyof TrainParams; label: string; step?: number }[] = [
  { key: 'accumulation_steps', label: 'Gradient accumulation steps' },
  { key: 'grad_clip_norm', label: 'Gradient clip norm (0 = off)', step: 0.1 },
  { key: 'weight_decay', label: 'Weight decay', step: 0.001 },
]

function initial(defaults: Record<string, unknown>): TrainParams {
  return defaults as unknown as TrainParams
}

/** Launch a training run on the loaded dataset. Server-side validation errors are shown beside the field. */
export function TrainDialog({ presets, datasetSource, datasetName, parentRunId, initialParams, initialPreset, onClose, onLaunched }: Props) {
  const startPreset = presets.presets.find(p => p.key === initialPreset) ?? presets.presets[0]
  const [preset, setPreset] = useState(startPreset.key)
  const [name, setName] = useState('')
  const [params, setParams] = useState<TrainParams>(() => ({ ...initial(startPreset.defaults), ...initialParams }))
  const [serverErrors, setServerErrors] = useState<FieldErrors>({})
  const [formError, setFormError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const selectedPreset = presets.presets.find(p => p.key === preset) ?? startPreset
  const isAgriTune = selectedPreset.engine === 'agritune'

  const clientErrors = useMemo(() => validateParams(params, presets.bounds), [params, presets.bounds])
  const errors = { ...serverErrors, ...clientErrors }
  const valid = Object.keys(clientErrors).length === 0

  const choosePreset = (key: string) => {
    setPreset(key)
    setParams(initial(presets.presets.find(p => p.key === key)!.defaults))
    setServerErrors({})
  }
  const set = <K extends keyof TrainParams>(key: K, value: TrainParams[K]) => {
    setParams(prev => ({ ...prev, [key]: value }))
    setServerErrors(prev => ({ ...prev, [key]: undefined }))
  }
  // Cached features are encoded once without augmentation and default to preloading into RAM.
  const setProvider = (provider: TrainParams['feature_provider']) => {
    setParams(prev => provider === 'cached'
      ? { ...prev, feature_provider: provider, augmentation_mode: 'none', feature_preload: prev.feature_preload ?? 'host' }
      : { ...prev, feature_provider: provider })
    setServerErrors(prev => ({ ...prev, feature_provider: undefined, feature_preload: undefined, augmentation_mode: undefined }))
  }
  const toggleTerm = (term: string) =>
    set('loss_terms', params.loss_terms.includes(term) ? params.loss_terms.filter(t => t !== term) : [...params.loss_terms, term])

  const submit = async () => {
    setBusy(true)
    setFormError(null)
    try {
      const run = await api.launchTraining({
        name: name.trim() || undefined, dataset_source: datasetSource, preset,
        params: launchParams(params, selectedPreset.engine), parent_run_id: parentRunId,
      })
      onLaunched(run.run_id)
    } catch (e) {
      const message = errorText(e)
      const field = /params\.([a-z_]+)(?:\.[a-z]+)?:/.exec(message)?.[1] as keyof TrainParams | undefined
      if (field) setServerErrors(prev => ({ ...prev, [field]: message.replace(/^.*?params\.[a-z_.]+:\s*/, '').replace(/["}\\]/g, '') }))
      else setFormError(message)
    } finally {
      setBusy(false)
    }
  }

  const inputStyle = { background: 'var(--c-raised)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }
  return (
    <div className="byod-overlay" role="dialog" aria-modal="true" aria-label="Train a model">
      <div className="byod-panel" style={{ width: 560 }}>
        <div className="byod-header">
          <h2 className="byod-title">Train a model</h2>
          <button className="byod-close" onClick={onClose} aria-label="Close"><X size={16} /></button>
        </div>
        <div className="byod-body" style={{ gap: 12, display: 'flex', flexDirection: 'column' }}>
          <label className="text-xs" style={{ color: 'var(--c-t2)' }}>Run name (optional)
            <input value={name} onChange={e => setName(e.target.value)} maxLength={100} className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={inputStyle} />
          </label>
          <div className="text-xs" style={{ color: 'var(--c-t2)' }}>Dataset
            <div className="px-2 py-1 rounded-lg mt-1 font-mono" style={{ ...inputStyle, opacity: 0.8 }} title={datasetSource}>{datasetName}</div>
          </div>
          <label className="text-xs" style={{ color: 'var(--c-t2)' }}>Preset
            <select value={preset} onChange={e => choosePreset(e.target.value)} className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={inputStyle}>
              {presets.presets.map(p => <option key={p.key} value={p.key}>{p.engine === 'agritune' ? 'AgriTune · ' : ''}{p.label}</option>)}
            </select>
            <span className="block mt-1" style={{ color: 'var(--c-t3)' }}>{selectedPreset.description}</span>
          </label>

          {isAgriTune && <div className="byod-hint" style={{ padding: '9px 10px', borderRadius: 8, background: 'var(--c-raised)' }}>
            AgriTune freezes the image encoder and trains only the selected decoder. With no encoder URL configured, the stack uses its deterministic fake encoder so the complete workflow remains testable offline.
          </div>}

          <div className="grid gap-x-3 gap-y-2 train-dialog-grid">
            <label className="text-xs" style={{ color: 'var(--c-t2)' }}>Input resolution
              <select value={params.input_resolution ?? 'image_size'} onChange={e => set('input_resolution', e.target.value as TrainParams['input_resolution'])}
                      aria-invalid={!!errors.input_resolution} className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={inputStyle}>
                <option value="image_size">Square, at Image size</option>
                <option value="native">Original size (all images alike)</option>
              </select>
              {errors.input_resolution && <span role="alert" className="block mt-0.5" style={{ color: 'var(--color-danger)' }}>{errors.input_resolution}</span>}
            </label>
            {FIELDS.map(f => {
              const b = presets.bounds[f.key]
              return (
                <label key={f.key} className="text-xs" style={{ color: 'var(--c-t2)' }}>{f.label}
                  <input type="number" value={Number.isNaN(params[f.key] as number) ? '' : (params[f.key] as number)}
                         min={b?.min} max={typeof b?.max === 'number' ? b.max : f.key === 'score_every' ? params.epochs : undefined}
                         step={f.step ?? 1} aria-invalid={!!errors[f.key]}
                         onChange={e => set(f.key, (e.target.value === '' ? NaN : Number(e.target.value)) as never)}
                         className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={{ ...inputStyle, borderColor: errors[f.key] ? 'var(--color-danger)' : 'var(--c-border)' }} />
                  {errors[f.key] && <span role="alert" className="block mt-0.5" style={{ color: 'var(--color-danger)' }}>{errors[f.key]}</span>}
                </label>
              )
            })}
          </div>

          {!isAgriTune && <label className="text-xs" style={{ color: 'var(--c-t2)' }}>Split
            <select value={params.split_mode} onChange={e => set('split_mode', e.target.value)} className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={inputStyle}>
              <option value="random">Random (by image name)</option>
              <option value="column">Use the dataset's split column</option>
            </select>
            {errors.split_mode && <span role="alert" style={{ color: 'var(--color-danger)' }}>{errors.split_mode}</span>}
          </label>}

          {!isAgriTune && <fieldset className="text-xs" style={{ color: 'var(--c-t2)' }}>
            <legend>Loss terms and weights</legend>
            <div className="flex gap-4 flex-wrap mt-1">
              {presets.loss_terms.map(t => (
                <label key={t} className="flex items-center gap-1.5">
                  <input type="checkbox" checked={params.loss_terms.includes(t)} onChange={() => toggleTerm(t)} /> {t}
                  <input type="number" aria-label={`Weight of ${t}`} min={0} max={10} step={0.1} disabled={!params.loss_terms.includes(t)}
                         value={params.loss_weights[t] ?? 1} onChange={e => set('loss_weights', { ...params.loss_weights, [t]: Number(e.target.value) })}
                         className="px-1.5 py-0.5 rounded-md text-xs" style={{ ...inputStyle, width: 58 }} />
                </label>
              ))}
            </div>
            {(errors.loss_terms || errors.loss_weights) && <span role="alert" style={{ color: 'var(--color-danger)' }}>{errors.loss_terms ?? errors.loss_weights}</span>}
          </fieldset>}

          {isAgriTune && <div className="grid gap-x-3 gap-y-2 train-dialog-grid">
            <label className="text-xs" style={{ color: 'var(--c-t2)' }}>Feature provider
              <select value={params.feature_provider ?? 'hybrid'} onChange={e => setProvider(e.target.value as TrainParams['feature_provider'])}
                      className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={inputStyle}>
                <option value="cached">Encode once, then train from the store (fastest)</option>
                <option value="hybrid">Hybrid cache (needed for augmentation)</option>
                <option value="online">Online every time</option>
              </select>
            </label>
            {params.feature_provider === 'cached' && <label className="text-xs" style={{ color: 'var(--c-t2)' }}>Preload features
              <select value={params.feature_preload ?? 'host'} onChange={e => set('feature_preload', e.target.value as TrainParams['feature_preload'])}
                      className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={inputStyle}>
                <option value="host">Into RAM (recommended)</option>
                <option value="device">Into GPU memory (must fit)</option>
                <option value="none">No, read from disk each epoch</option>
              </select>
            </label>}
            <label className="text-xs" style={{ color: 'var(--c-t2)' }}>Image augmentation
              <select value={params.augmentation_mode ?? 'none'} onChange={e => set('augmentation_mode', e.target.value as TrainParams['augmentation_mode'])}
                      disabled={params.feature_provider === 'cached'} aria-invalid={!!errors.augmentation_mode}
                      title={params.feature_provider === 'cached' ? 'Cached features are encoded once without augmentation; choose the hybrid provider to augment' : undefined}
                      className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={inputStyle}>
                <option value="none">None</option>
                <option value="offline">Fixed augmented samples</option>
                <option value="online">Fresh every epoch</option>
                <option value="hybrid">Hybrid</option>
              </select>
            </label>
            {AGRITUNE_NUMBERS.map(f => (
              <label key={f.key} className="text-xs" style={{ color: 'var(--c-t2)' }}>{f.label}
                <input type="number" value={params[f.key] === undefined || Number.isNaN(params[f.key] as number) ? '' : (params[f.key] as number)}
                       min={presets.bounds[f.key]?.min} max={presets.bounds[f.key]?.max as number | undefined} step={f.step ?? 1}
                       aria-invalid={!!errors[f.key]}
                       onChange={e => set(f.key, (e.target.value === '' ? NaN : Number(e.target.value)) as never)}
                       className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={{ ...inputStyle, borderColor: errors[f.key] ? 'var(--color-danger)' : 'var(--c-border)' }} />
                {errors[f.key] && <span role="alert" className="block mt-0.5" style={{ color: 'var(--color-danger)' }}>{errors[f.key]}</span>}
              </label>
            ))}
            <label className="text-xs" style={{ color: 'var(--c-t2)' }}>Loss objective
              <select value={params.agritune_loss ?? 'ce_dice'} onChange={e => set('agritune_loss', e.target.value as TrainParams['agritune_loss'])}
                      className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={inputStyle}>
                <option value="ce">Cross entropy</option>
                <option value="bce">Binary cross entropy</option>
                <option value="dice">Dice</option>
                <option value="ce_dice">Cross entropy + Dice</option>
                <option value="bce_dice">Binary cross entropy + Dice</option>
              </select>
            </label>
            <div className="text-xs" style={{ color: 'var(--c-t3)', alignSelf: 'end', paddingBottom: 5 }}>
              Total and component losses are saved per image and appear automatically as Explorer axes.
            </div>
          </div>}

          <div className="grid gap-x-3 gap-y-2 train-dialog-grid">
            {!isAgriTune && <label className="text-xs" style={{ color: 'var(--c-t2)' }}>Loader processes (blank = automatic)
              <input type="number" min={presets.bounds.num_workers?.min} max={presets.bounds.num_workers?.max as number | undefined} step={1}
                     value={params.num_workers ?? ''} aria-invalid={!!errors.num_workers}
                     onChange={e => set('num_workers', e.target.value === '' ? undefined : Number(e.target.value))}
                     className="w-full px-2 py-1 rounded-lg text-xs mt-1" style={{ ...inputStyle, borderColor: errors.num_workers ? 'var(--color-danger)' : 'var(--c-border)' }} />
              {errors.num_workers && <span role="alert" className="block mt-0.5" style={{ color: 'var(--color-danger)' }}>{errors.num_workers}</span>}
            </label>}
            <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--c-t2)' }}>
              <input type="checkbox" checked={params.amp ?? true} onChange={e => set('amp', e.target.checked)} /> Mixed precision (GPU only)
            </label>
          </div>

          {!isAgriTune && <label className="flex items-center gap-2 text-xs" style={{ color: 'var(--c-t2)' }}>
            <input type="checkbox" checked={params.pretrained} onChange={e => set('pretrained', e.target.checked)} /> Start from ImageNet-pretrained weights (where the preset has a backbone)
          </label>}

          {formError && <div role="alert" className="byod-error">{formError}</div>}
          <div className="flex justify-end gap-2">
            <button className="btn-ghost px-3 py-1 text-xs" onClick={onClose}>Cancel</button>
            <button className="btn-primary px-3 py-1 text-xs flex items-center gap-1.5" disabled={!valid || busy} onClick={() => void submit()}
                    style={{ opacity: !valid || busy ? 0.6 : 1 }}>
              {busy && <Loader2 size={12} className="byod-spin" />} Start training
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
