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
  return errors
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

/** "epoch 3/20" while running; "queued" while scheduled. */
export function progressLabel(status: string, epochsDone: number, totalEpochs: number): string {
  if (status === 'SCHEDULED') return 'queued'
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
  if (params.loss_terms) {
    out.loss_terms = params.loss_terms.split(',').filter(Boolean)
    out.loss_weights = Object.fromEntries(
      out.loss_terms.filter(t => Number.isFinite(Number(params[`weight_${t}`])) && params[`weight_${t}`] !== undefined)
        .map(t => [t, Number(params[`weight_${t}`])]),
    )
  }
  return out
}
