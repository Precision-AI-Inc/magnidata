import type { Note } from './types'

import { batches, METRIC_KEYS_PER_REQUEST } from './lib/metrics'

const BASE = '/api'

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) {
    const body = await r.text()
    throw new ApiError(`${r.status}: ${body}`, r.status)
  }
  return r.json()
}

async function mutate<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body != null ? JSON.stringify(body) : undefined,
  })
  if (!r.ok && r.status !== 204) {
    const text = await r.text()
    throw new Error(`${r.status}: ${text}`)
  }
  if (r.status === 204) return undefined as T
  return r.json()
}

async function postForm<T>(
  url: string,
  formData: FormData,
  onUploadProgress?: (percent: number) => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', url)
    xhr.upload.onprogress = event => {
      if (event.lengthComputable && event.total > 0) {
        onUploadProgress?.(Math.round((event.loaded / event.total) * 100))
      }
    }
    xhr.onload = () => {
      const body = xhr.responseText || ''
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(`${xhr.status}: ${errorBody(body)}`))
        return
      }
      try {
        resolve(JSON.parse(body) as T)
      } catch (e) {
        reject(e instanceof Error ? e : new Error(String(e)))
      }
    }
    xhr.onerror = () => reject(new Error('Network error while uploading dataset'))
    xhr.onabort = () => reject(new Error('Upload aborted'))
    xhr.send(formData)
  })
}

function errorBody(body: string): string {
  try {
    const parsed = JSON.parse(body) as { error?: string }
    return parsed.error || body
  } catch {
    return body
  }
}

/** User-facing text for a thrown API error — drops the "409: " status prefix the helpers
 * above add, which means nothing to a user. */
export function errorText(e: unknown): string {
  return (e instanceof Error ? e.message : String(e)).replace(/^\d{3}: /, '')
}

export interface DatasetMeta {
  name: string
  description: string
  source: string
  file_size_bytes?: number
  has_embeddings?: boolean
  is_lfs_pointer?: boolean
  lfs_size_bytes?: number
  id?: number               // present for user-created datasets
  deletable?: boolean
  parent_source?: string
  parent_sources?: string[]
  row_count?: number
}

export interface NumericColumnSummary {
  name: string
  min: number
  max: number
  mean: number
  histogram: number[]
}

/** Per-dataset statistics for the workspace detail panel. */
export interface DatasetSummary {
  row_count: number
  column_count: number
  numeric_column_count: number
  numeric_columns: NumericColumnSummary[]
  image_column: string | null
  sample_images: string[]
  mask_coverage: { checked: number; with_masks: number }
  partitions_committed: number
  has_embeddings: boolean
  truncated: boolean
}

export interface MergeDatasetPayload {
  name: string
  description?: string
  sources: string[]
}

/** Run counts per dataset source from MLflow; datasets with no runs are absent. */
export type ExperimentCounts = Record<string, { total: number; active: number }>

/** { embeddings: { "<file_path>": number[] } } */
export interface EmbeddingsFile {
  embeddings: Record<string, number[]>
}

/** A comparison embeddings file: <stem>_<variant>.json (variant = model name) */
export interface EmbeddingVariant {
  name: string
  variant: string
}

export interface CreateDatasetPayload {
  name: string
  description?: string
  parent_source: string
  image_names: string[]
  query_image_names?: string[]
  complement_top_k?: number
  cloud_source?: string
}

/** A row-level partition annotation, keyed by image filename. cluster_id is 1-based (null = unassigned). */
export interface Partition {
  image_name: string
  cluster_id: number | null
}

/** One entry in the "Load Demo" registry — not every demo is buildable yet. */
export interface DemoEntry {
  key: string
  name: string
  description: string
  enabled: boolean
}

export interface BuildLimits {
  max_images: number
  max_upload_bytes: number
  max_embeddings_bytes: number
  max_request_bytes: number
}

export interface LocalFolder {
  folder: string
  image_count: number
  label_count: number
  built: boolean
}

export interface BuildJobStatus {
  status: 'uploading' | 'queued' | 'staging' | 'extracting_features' | 'registering' | 'done' | 'error'
  percent: number
  message: string
  name?: string             // dataset being built, e.g. "AgriStress-500"
  dataset?: DatasetMeta
  error?: string
}

export interface EmbeddingAnalysisStatus {
  job_id: string
  status: 'queued' | 'running' | 'done' | 'error'
  progress: number
  stage: string
  k?: number
  error?: string
  labels?: number[]
  projections?: Record<string, number[][]>
}


// ── MLflow experiments ────────────────────────────────────────────────────────
export type RunStatus = 'SCHEDULED' | 'RUNNING' | 'FINISHED' | 'FAILED' | 'KILLED'

export interface MlflowStatus {
  reachable: boolean
  tracking_uri: string | null
  ui_url: string | null
  version: string | null
  error: string | null
}

export interface Experiment {
  experiment_id: string
  name: string
  artifact_location: string
  lifecycle_stage: string
  creation_time: number
  last_update_time: number
  tags: Record<string, string>
}

export interface RunSummary {
  run_id: string
  name: string | null
  experiment_id: string
  status: RunStatus
  start_time: number
  end_time: number | null
  duration_ms: number | null
  user: string | null
  metrics: Record<string, number>
  params: Record<string, string>
  magnidata: Record<string, string>
  has_per_sample: boolean
  has_split: boolean
  mlflow_ui_url: string | null
}

export interface RunsPage {
  runs: RunSummary[]
  next_page_token: string | null
}

export interface RunSearch {
  experimentIds: string[]
  filter?: string
  orderBy?: string
  maxResults?: number
  pageToken?: string | null
}

export interface RunDetailData extends RunSummary {
  tags: Record<string, string>
  metric_keys: string[]
  scored_epochs: number[]
  artifact_uri: string
  artifacts: ArtifactEntry[]
}

export interface TrainingPresets {
  presets: { key: string; label: string; description: string; engine: 'reference' | 'agritune'; defaults: Record<string, unknown> }[]
  bounds: Record<string, import('./lib/train').Bound>
  loss_terms: string[]
  backend: { available: boolean; reason: string | null; worker_heartbeat: number | null }
  mlflow: MlflowStatus
}

export interface LaunchBody {
  name?: string
  dataset_source: string
  preset: string
  params: Record<string, unknown>
  parent_run_id?: string
}

export interface SampleHistory {
  image_name: string
  split: string
  epochs: number[]
  terms: Record<string, (number | null)[]>
  iou?: (number | null)[]
  confidence?: (number | null)[]
  prediction_epochs: number[]
}

export interface InferenceStatus {
  status: 'idle' | 'queued' | 'running' | 'ready' | 'error'
  image: string
  epoch: number
  error?: string
}

export interface RunMatch {
  matched: number
  run_only: number
  dataset_only: number
  run_only_examples: string[]
  dataset_only_examples: string[]
}

export interface MetricHistory { step: number; value: number; timestamp: number }

export interface ArtifactEntry { path: string; is_dir: boolean; file_size: number | null }

/** Thrown for API errors so callers can tell "MLflow is down" (503) from the rest. */
export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message) }
}

export const api = {
  health: () => get<{ status: string; local: { datalake_accessible: boolean; root: string } }>(`${BASE}/health`),

  datasets: () => get<DatasetMeta[]>(`${BASE}/datasets`),
  createDataset: (payload: CreateDatasetPayload) => mutate<DatasetMeta>('POST', `${BASE}/datasets`, payload),
  deleteDataset: (id: number) => mutate<void>('DELETE', `${BASE}/datasets/${id}`),

  // ── Build a dataset from images (BYOD) ──────────────────────────────────────
  listDemos: () => get<DemoEntry[]>(`${BASE}/datasets/demos`),
  buildLimits: () => get<BuildLimits>(`${BASE}/datasets/build/limits`),
  buildDataset: (formData: FormData, onUploadProgress?: (percent: number) => void) =>
    postForm<{ job_id: string }>(`${BASE}/datasets/build`, formData, onUploadProgress),
  listLocalFolders: () => get<LocalFolder[]>(`${BASE}/datasets/build/local-folders`),
  buildLocal: (folder: string, name?: string, description?: string) => {
    const fd = new FormData()
    fd.append('source_type', 'local')
    fd.append('folder', folder)
    if (name) fd.append('name', name)
    if (description) fd.append('description', description)
    return postForm<{ job_id: string }>(`${BASE}/datasets/build`, fd)
  },
  buildDemo: (demoKey: string) => {
    const fd = new FormData()
    fd.append('source_type', 'demo')
    fd.append('demo_key', demoKey)
    return postForm<{ job_id: string }>(`${BASE}/datasets/build`, fd)
  },
  getBuildJob: (jobId: string) => get<BuildJobStatus>(`${BASE}/datasets/build/${jobId}`),
  datasetCsvUrl: (source: string) => `${BASE}/dataset/csv?source=${encodeURIComponent(source)}`,
  datasetEmbeddingsUrl: (source: string, variant?: string) =>
    `${BASE}/dataset/embeddings?source=${encodeURIComponent(source)}${variant ? `&variant=${encodeURIComponent(variant)}` : ''}`,
  getEmbeddings: (source: string, variant?: string) =>
    get<EmbeddingsFile>(api.datasetEmbeddingsUrl(source, variant)),
  datasetEmbeddingVariants: (source: string) =>
    get<EmbeddingVariant[]>(`${BASE}/dataset/embeddings/variants?source=${encodeURIComponent(source)}`),

  // ── Server-side embedding compute (browser receives only the small results) ──
  embeddingProjection: (source: string, method: string, variant?: string) =>
    get<{ positions: number[][] }>(
      `${BASE}/embedding/projection?source=${encodeURIComponent(source)}&method=${method}${variant ? `&variant=${encodeURIComponent(variant)}` : ''}`),
  embeddingClusters: (source: string, k: number, variant?: string) =>
    get<{ labels: number[] }>(
      `${BASE}/embedding/clusters?source=${encodeURIComponent(source)}&k=${k}${variant ? `&variant=${encodeURIComponent(variant)}` : ''}`),
  startEmbeddingAnalysis: (source: string, k: number, reduction = 'pca', variant?: string) =>
    mutate<{ job_id: string }>('POST', `${BASE}/embedding/analysis`, { source, k, reduction, ...(variant ? { variant } : {}) }),
  getEmbeddingAnalysis: (jobId: string) =>
    get<EmbeddingAnalysisStatus>(`${BASE}/embedding/analysis/${encodeURIComponent(jobId)}`),
  embeddingCompare: (source: string, variant: string) =>
    get<{ nbr: number[][]; w: number[][]; origIdx: number[] }>(
      `${BASE}/embedding/compare?source=${encodeURIComponent(source)}&variant=${encodeURIComponent(variant)}`),


  /** Returns a URL string suitable for <img src=...> — fetch happens in browser */
  imageUrl:  (path: string, maxSize = 720) =>
    `${BASE}/image?path=${encodeURIComponent(path)}&max_size=${maxSize}`,

  maskUrl:   (path: string, maxSize = 720) =>
    `${BASE}/mask?path=${encodeURIComponent(path)}&max_size=${maxSize}`,

  overlayUrl: (img: string, mask: string, alpha = 0.45, maxSize = 720) =>
    `${BASE}/overlay?img=${encodeURIComponent(img)}&mask=${encodeURIComponent(mask)}&alpha=${alpha}&max_size=${maxSize}`,

  /** Segmentation mask rasterized on the fly from the image's COCO label file
   * (transparent PNG) — 404 if the image has no annotations. */
  annotationMaskUrl: (path: string, maxSize = 1280) =>
    `${BASE}/annotation/mask?path=${encodeURIComponent(path)}&max_size=${maxSize}`,

  /** The image composited with its segmentation mask at the given opacity (0..1). */
  annotationOverlayUrl: (path: string, alpha: number, maxSize = 1280) =>
    `${BASE}/annotation/overlay?path=${encodeURIComponent(path)}&alpha=${alpha}&max_size=${maxSize}`,

  // ── Cluster partitions (cluster_id, persisted per dataset by image filename) ──
  mergeDatasets: (payload: MergeDatasetPayload) =>
    mutate<DatasetMeta>('POST', `${BASE}/datasets/merge`, payload),
  datasetExperimentCounts: () => get<ExperimentCounts>(`${BASE}/datasets/experiment-counts`),
  datasetSummary: (source: string) =>
    get<DatasetSummary>(`${BASE}/datasets/summary?source=${encodeURIComponent(source)}`),
  getPartitions: (source: string) =>
    get<{ partitions: Partition[] }>(`${BASE}/partitions?source=${encodeURIComponent(source)}`),
  commitPartitions: (source: string, entries: { image_name: string; cluster_id: number }[]) =>
    mutate<{ committed: number }>('POST', `${BASE}/partitions/commit`, { source, entries }),


  // ── MLflow experiments ──────────────────────────────────────────────────────
  getMlflowStatus: () => get<MlflowStatus>(`${BASE}/mlflow/status`),
  listExperiments: (query = '') =>
    get<{ experiments: Experiment[]; next_page_token: string | null }>(
      `${BASE}/experiments?max_results=200${query ? `&query=${encodeURIComponent(query)}` : ''}`),
  searchRuns: (p: RunSearch) => {
    const q = new URLSearchParams()
    if (p.experimentIds.length) q.set('experiment_ids', p.experimentIds.join(','))
    if (p.filter) q.set('filter', p.filter)
    if (p.orderBy) q.set('order_by', p.orderBy)
    q.set('max_results', String(p.maxResults ?? 50))
    if (p.pageToken) q.set('page_token', p.pageToken)
    return get<RunsPage>(`${BASE}/runs?${q}`)
  },

  getTrainingPresets: () => get<TrainingPresets>(`${BASE}/training/presets`),
  launchTraining: (body: LaunchBody) => mutate<RunSummary>('POST', `${BASE}/training/runs`, body),
  cancelRun: (runId: string) =>
    mutate<{ run_id: string; status: string; cancel_requested: boolean }>('POST', `${BASE}/runs/${runId}/cancel`),
  getRun: (runId: string) => get<RunDetailData>(`${BASE}/runs/${runId}`),
  getRunMetrics: (runId: string, keys: string[]) =>
    get<{ metrics: Record<string, MetricHistory[]> }>(
      `${BASE}/runs/${runId}/metrics?keys=${encodeURIComponent(keys.join(','))}`),
  getRunsMetrics: (runIds: string[], keys: string[]) =>
    mutate<{ runs: Record<string, Record<string, MetricHistory[]>> }>('POST', `${BASE}/runs/metrics`, { run_ids: runIds, keys }),
  /** Every key's history, in as many requests as the API's per-request key limit needs. AgriTune logs ~140 keys. */
  getAllRunMetrics: async (runId: string, keys: string[]) => {
    const parts = await Promise.all(batches(keys, METRIC_KEYS_PER_REQUEST).map(part => api.getRunMetrics(runId, part)))
    return { metrics: Object.assign({}, ...parts.map(part => part.metrics)) as Record<string, MetricHistory[]> }
  },
  getAllRunsMetrics: async (runIds: string[], keys: string[]) => {
    const parts = await Promise.all(batches(keys, METRIC_KEYS_PER_REQUEST).map(part => api.getRunsMetrics(runIds, part)))
    const runs: Record<string, Record<string, MetricHistory[]>> = {}
    for (const part of parts) for (const [id, metrics] of Object.entries(part.runs)) runs[id] = { ...runs[id], ...metrics }
    return { runs }
  },
  /** Derived per-sample columns of a run at an epoch, as CSV text (`image_name` first). */
  getRunColumns: async (runId: string, epoch: number | 'latest' = 'latest') => {
    const r = await fetch(`${BASE}/runs/${runId}/columns?epoch=${epoch}`)
    if (!r.ok) throw new ApiError(`${r.status}: ${await r.text()}`, r.status)
    return r.text()
  },
  getRunSample: (runId: string, imageName: string) =>
    get<SampleHistory>(`${BASE}/runs/${runId}/samples/${encodeURIComponent(imageName)}`),
  getInferenceStatus: (runId: string, imageName: string) =>
    get<InferenceStatus>(`${BASE}/runs/${runId}/inference?image=${encodeURIComponent(imageName)}`),
  runInference: (runId: string, imageName: string) =>
    mutate<InferenceStatus>('POST', `${BASE}/runs/${runId}/inference`, { image: imageName }),
  runPredictionUrl: (runId: string, imageName: string, epoch: number | 'latest', alpha: number, maxSize = 1280) =>
    `${BASE}/runs/${runId}/prediction?image=${encodeURIComponent(imageName)}&epoch=${epoch}&alpha=${alpha}&max_size=${maxSize}`,
  /** Per-image comparison of two runs (hardness and loss change from a to b), as CSV text. */
  getCompareSamples: async (a: string, b: string, epoch: number | 'latest' = 'latest') => {
    const r = await fetch(`${BASE}/runs/compare-samples?a=${a}&b=${b}&epoch=${epoch}`)
    if (!r.ok) throw new ApiError(`${r.status}: ${await r.text()}`, r.status)
    return r.text()
  },
  getRunMatch: (runId: string, datasetSource: string) =>
    get<RunMatch>(`${BASE}/runs/${runId}/match?dataset=${encodeURIComponent(datasetSource)}`),
  linkRun: (runId: string, datasetSource: string) =>
    mutate<RunSummary>('POST', `${BASE}/runs/${runId}/link`, { dataset_source: datasetSource }),
  listRunArtifacts: (runId: string, path = '') =>
    get<{ artifacts: ArtifactEntry[] }>(`${BASE}/runs/${runId}/artifacts${path ? `?path=${encodeURIComponent(path)}` : ''}`),
  runArtifactUrl: (runId: string, path: string) =>
    `${BASE}/runs/${runId}/artifact?path=${encodeURIComponent(path)}`,

  // Notes
  getNotes:    (stem?: string) =>
    get<Note[]>(`${BASE}/notes${stem ? `?stem=${encodeURIComponent(stem)}` : ''}`),

  createNote:  (stem: string, image_path: string, note: string) =>
    mutate<Note>('POST', `${BASE}/notes`, { stem, image_path, note }),

  updateNote:  (id: number, note: string) =>
    mutate<Note>('PUT', `${BASE}/notes/${id}`, { note }),

  deleteNote:  (id: number) =>
    mutate<void>('DELETE', `${BASE}/notes/${id}`),
}
