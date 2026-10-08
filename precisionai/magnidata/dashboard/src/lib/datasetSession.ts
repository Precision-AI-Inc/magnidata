import type { DatasetMeta } from '../api'

export const LAST_DATASET_SOURCE_KEY = 'magnidata.lastDatasetSource'

type ReadStorage = Pick<Storage, 'getItem'>
type WriteStorage = Pick<Storage, 'setItem' | 'removeItem'>

export function readLastDatasetSource(storage: ReadStorage = localStorage): string | null {
  try {
    const source = storage.getItem(LAST_DATASET_SOURCE_KEY)?.trim()
    return source || null
  } catch {
    return null
  }
}

export function rememberDatasetSource(source: string, storage: WriteStorage = localStorage): void {
  try {
    storage.setItem(LAST_DATASET_SOURCE_KEY, source)
  } catch {
    // Storage may be disabled; dataset loading must still work for this session.
  }
}

export function forgetDatasetSource(source: string, storage: ReadStorage & WriteStorage = localStorage): void {
  try {
    if (storage.getItem(LAST_DATASET_SOURCE_KEY) === source) storage.removeItem(LAST_DATASET_SOURCE_KEY)
  } catch {
    // Storage may be disabled.
  }
}

export function rememberedDataset(datasets: DatasetMeta[], source: string | null): DatasetMeta | null {
  return source ? datasets.find(dataset => dataset.source === source) ?? null : null
}

export async function retry<T>(operation: () => Promise<T>, delays = [0, 500, 1500]): Promise<T> {
  let error: unknown
  for (const delay of delays) {
    if (delay > 0) await new Promise(resolve => window.setTimeout(resolve, delay))
    try {
      return await operation()
    } catch (caught) {
      error = caught
    }
  }
  throw error
}
