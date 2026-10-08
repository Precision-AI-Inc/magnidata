import { describe, expect, it, vi } from 'vitest'
import type { DatasetMeta } from '../api'
import {
  LAST_DATASET_SOURCE_KEY, forgetDatasetSource, readLastDatasetSource,
  rememberedDataset, rememberDatasetSource, retry,
} from './datasetSession'

function storage() {
  const values = new Map<string, string>()
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value) },
    removeItem: (key: string) => { values.delete(key) },
  }
}

describe('dataset session memory', () => {
  it('remembers and conditionally forgets a dataset source', () => {
    const store = storage()
    rememberDatasetSource('data_user/latest.csv', store)
    expect(readLastDatasetSource(store)).toBe('data_user/latest.csv')
    forgetDatasetSource('data_user/other.csv', store)
    expect(store.getItem(LAST_DATASET_SOURCE_KEY)).toBe('data_user/latest.csv')
    forgetDatasetSource('data_user/latest.csv', store)
    expect(readLastDatasetSource(store)).toBeNull()
  })

  it('matches a remembered source against the current catalogue', () => {
    const datasets = [{ name: 'Latest', source: 'data_user/latest.csv', description: '' }] as DatasetMeta[]
    expect(rememberedDataset(datasets, 'data_user/latest.csv')).toBe(datasets[0])
    expect(rememberedDataset(datasets, 'data_user/missing.csv')).toBeNull()
  })
})

describe('retry', () => {
  it('returns after a transient failure', async () => {
    const operation = vi.fn()
      .mockRejectedValueOnce(new Error('busy'))
      .mockResolvedValue('ready')
    await expect(retry(operation, [0, 0])).resolves.toBe('ready')
    expect(operation).toHaveBeenCalledTimes(2)
  })

  it('throws the final failure', async () => {
    const operation = vi.fn().mockRejectedValueOnce(new Error('one')).mockRejectedValueOnce(new Error('two'))
    await expect(retry(operation, [0, 0])).rejects.toThrow('two')
  })
})
