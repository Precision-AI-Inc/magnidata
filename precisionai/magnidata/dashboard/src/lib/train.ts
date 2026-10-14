export interface Bound {
  type: 'int' | 'float' | 'choice' | 'list' | 'map' | 'bool'
  min?: number
  /** A number, or the name of another parameter (`score_every` may not exceed `epochs`). */
  max?: number | string
  multiple_of?: number
  choices?: string[]
  min_items?: number
}

export interface TrainParams {
  epochs: number
  score_every: number
  batch_size: number
  image_size: number
  lr: number
  val_fraction: number
  seed: number
  split_mode: string
  loss_terms: string[]
  loss_weights: Record<string, number>
  top_k_predictions: number
  pretrained: boolean
  amp: boolean
  /** Data-loader processes; unset lets the trainer choose from its CPU count. */
  num_workers?: number
  feature_provider?: 'hybrid' | 'online'
  augmentation_mode?: 'none' | 'offline' | 'online' | 'hybrid'
  agritune_loss?: 'ce' | 'bce' | 'dice' | 'ce_dice' | 'bce_dice'
}

export type FieldErrors = Partial<Record<keyof TrainParams, string>>

const NUMERIC: (keyof TrainParams)[] = [
  'epochs', 'batch_size', 'image_size', 'seed', 'top_k_predictions', 'lr', 'val_fraction', 'score_every',
]

function range(bound: Bound, params: TrainParams): [number, number] {
  const max = typeof bound.max === 'string' ? Number(params[bound.max as keyof TrainParams]) : (bound.max ?? Infinity)
  return [bound.min ?? -Infinity, max]
}

/** Client-side mirror of the API's bounds; returns one message per invalid field (empty when valid). */
export function validateParams(params: TrainParams, bounds: Record<string, Bound>): FieldErrors {
  const errors: FieldErrors = {}
  for (const field of NUMERIC) {
    const bound = bounds[field]
    const value = params[field] as number
    if (!bound) continue
    if (typeof value !== 'number' || !Number.isFinite(value)) { errors[field] = 'must be a number'; continue }
    if (bound.type === 'int' && !Number.isInteger(value)) { errors[field] = 'must be a whole number'; continue }
    const [lo, hi] = range(bound, params)
    if (value < lo || value > hi) errors[field] = `must be between ${lo} and ${hi}`
    else if (bound.multiple_of && value % bound.multiple_of) errors[field] = `must be a multiple of ${bound.multiple_of}`
  }
  const workers = bounds.num_workers
  if (params.num_workers !== undefined && workers) {
    const n = params.num_workers
    if (!Number.isInteger(n)) errors.num_workers = 'must be a whole number'
    else if (n < (workers.min ?? 0) || n > Number(workers.max ?? 16)) errors.num_workers = `must be between ${workers.min ?? 0} and ${workers.max ?? 16}`
  }
  const split = bounds.split_mode
  if (split?.choices && !split.choices.includes(params.split_mode)) errors.split_mode = `must be one of ${split.choices.join(', ')}`
  const terms = bounds.loss_terms
  if (params.loss_terms.length < (terms?.min_items ?? 1)) errors.loss_terms = 'choose at least one loss term'
  else if (terms?.choices && params.loss_terms.some(t => !terms.choices!.includes(t))) errors.loss_terms = 'unknown loss term'
  const weights = bounds.loss_weights
  for (const [term, weight] of Object.entries(params.loss_weights)) {
    if (!params.loss_terms.includes(term)) continue
    if (!Number.isFinite(weight) || weight < (weights?.min ?? 0) || weight > Number(weights?.max ?? 10)) {
      errors.loss_weights = `weights must be between ${weights?.min ?? 0} and ${weights?.max ?? 10}`
    }
  }
  for (const field of ['feature_provider', 'augmentation_mode', 'agritune_loss'] as const) {
    const value = params[field]
    const choices = bounds[field]?.choices
    if (value !== undefined && choices && !choices.includes(value)) errors[field] = `must be one of ${choices.join(', ')}`
  }
  return errors
}

/** Settings the AgriTune engine does not use; the API rejects them, so they are never sent for its presets. */
const REFERENCE_ONLY: (keyof TrainParams)[] = ['loss_terms', 'num_workers', 'split_mode', 'pretrained']

/** The parameters to POST for a launch: the dialog state minus anything the preset's engine would reject. */
export function launchParams(params: TrainParams, engine: string): Partial<TrainParams> {
  const out: Partial<TrainParams> = { ...params }
  if (engine === 'agritune') for (const key of REFERENCE_ONLY) delete out[key]
  return out
}

export function isActive(status: string | null | undefined): boolean {
  return status === 'SCHEDULED' || status === 'RUNNING'
}

export interface TrainState {
  hasDataset: boolean
  mlflowReachable: boolean
  workerAvailable: boolean
  workerReason?: string | null
  activeRunId: string | null
}

/** Why "Train" is disabled, or null when it can be used. Earlier causes take precedence. */
export function trainDisabledReason(s: TrainState): string | null {
  if (!s.hasDataset) return 'Load a dataset to train on'
  if (!s.mlflowReachable) return 'MLflow is not reachable'
  if (!s.workerAvailable) return s.workerReason ? `Trainer worker unavailable: ${s.workerReason}` : 'Trainer worker unavailable'
  if (s.activeRunId) return 'Another training run is scheduled or running'
  return null
}

/** Current preparation/batch stage when available, with an epoch fallback for older trainers. */
export function progressLabel(
  status: string,
  epochsDone: number,
  totalEpochs: number,
  tags: Record<string, string> = {},
): string {
  if (status === 'SCHEDULED') return 'queued'
  const stage = tags['magnidata.progress_stage']
  const current = Number(tags['magnidata.progress_current'])
  const total = Number(tags['magnidata.progress_total'])
  if (stage) {
    const percent = Number.isFinite(current) && Number.isFinite(total) && total > 0
      ? ` · ${Math.min(100, Math.round((current / total) * 100))}%`
      : ''
    return `${stage}${percent}`
  }
  return `epoch ${Math.min(epochsDone + 1, totalEpochs || epochsDone + 1)}/${totalEpochs || '?'}`
}


const SETTINGS: (keyof TrainParams)[] = [
  'epochs', 'score_every', 'batch_size', 'image_size', 'lr', 'val_fraction', 'seed', 'top_k_predictions',
]

/** Training settings of an earlier run (MLflow param strings), to preload the Train dialog for a retrain. */
export function paramsFromRun(params: Record<string, string>): Partial<TrainParams> {
  const out: Partial<TrainParams> = {}
  for (const key of SETTINGS) {
    const value = Number(params[key])
    if (params[key] !== undefined && Number.isFinite(value)) (out as Record<string, unknown>)[key] = value
  }
  if (params.split_mode) out.split_mode = params.split_mode
  if (params.pretrained) out.pretrained = params.pretrained === 'true'
  if (params.amp) out.amp = params.amp === 'true'
  if (params.num_workers !== undefined && Number.isInteger(Number(params.num_workers))) out.num_workers = Number(params.num_workers)
  if (params.feature_provider === 'hybrid' || params.feature_provider === 'online') out.feature_provider = params.feature_provider
  if (params.augmentation_mode && ['none', 'offline', 'online', 'hybrid'].includes(params.augmentation_mode)) out.augmentation_mode = params.augmentation_mode as TrainParams['augmentation_mode']
  if (params.agritune_loss && ['ce', 'bce', 'dice', 'ce_dice', 'bce_dice'].includes(params.agritune_loss)) out.agritune_loss = params.agritune_loss as TrainParams['agritune_loss']
  if (params.loss_terms) {
    out.loss_terms = params.loss_terms.split(',').filter(Boolean)
    out.loss_weights = Object.fromEntries(
      out.loss_terms.filter(t => Number.isFinite(Number(params[`weight_${t}`])) && params[`weight_${t}`] !== undefined)
        .map(t => [t, Number(params[`weight_${t}`])]),
    )
  }
  return out
}
