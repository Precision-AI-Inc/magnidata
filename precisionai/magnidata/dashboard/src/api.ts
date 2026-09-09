import type { Note } from './types'

const BASE = '/api'

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) {
    const body = await r.text()
    throw new Error(`${r.status}: ${body}`)
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
  row_count?: number
}

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
  dataset?: DatasetMeta
  error?: string
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
  getPartitions: (source: string) =>
    get<{ partitions: Partition[] }>(`${BASE}/partitions?source=${encodeURIComponent(source)}`),
  commitPartitions: (source: string, entries: { image_name: string; cluster_id: number }[]) =>
    mutate<{ committed: number }>('POST', `${BASE}/partitions/commit`, { source, entries }),

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
