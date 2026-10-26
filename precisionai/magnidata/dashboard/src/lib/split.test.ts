import { describe, expect, it } from 'vitest'
import type { CSVRow } from '../types'
import {
  defaultTrainingCategorical, partitionColumns, partitionEligible, splitColor, splitCounts, splitSummary, SPLIT_COLORS,
} from './split'

const rows = (values: unknown[], col = 'split'): CSVRow[] => values.map((v, i) => ({ _idx: i, [col]: v }) as CSVRow)

describe('split palette', () => {
  it('maps train, val and test to fixed distinct colours', () => {
    expect(splitColor('train')).toBe(SPLIT_COLORS.train)
    expect(new Set([splitColor('train'), splitColor('val'), splitColor('test')]).size).toBe(3)
    expect(splitColor('other')).not.toBe(splitColor('train'))
  })
})

describe('split counts', () => {
  const data = rows(['val', 'train', 'train', null, '', 'train'])
  it('counts rows per split in train/val order and skips unlabelled rows', () => {
    expect(splitCounts(data)).toEqual([['train', 3], ['val', 1]])
    expect(splitSummary(data)).toBe('train 3 · val 1')
  })
})

describe('partition eligibility', () => {
  it('needs 2 to 12 distinct values', () => {
    expect(partitionEligible('split', rows(['train', 'val', 'train']))).toBe(true)
    expect(partitionEligible('split', rows(['train', 'train']))).toBe(false)
    expect(partitionEligible('c', rows(Array.from({ length: 13 }, (_, i) => `v${i}`), 'c'))).toBe(false)
    expect(partitionEligible('c', rows(Array.from({ length: 12 }, (_, i) => `v${i}`), 'c'))).toBe(true)
  })
  it('always offers the dataset\'s own cluster column, even when numeric', () => {
    expect(partitionColumns(['split'], rows(['train', 'val']), 'cluster')).toEqual(['split', 'cluster'])
  })
  it('lists split first, then the cluster column', () => {
    const data = rows(['a', 'b']).map(r => ({ ...r, split: 'train', cluster: 'x', camera: r.split })) as CSVRow[]
    data[1].split = 'val'; data[1].cluster = 'y'
    expect(partitionColumns(['camera', 'cluster', 'split'], data, 'cluster')).toEqual(['split', 'cluster', 'camera'])
  })
})

describe('defaultTrainingCategorical', () => {
  it('adds split once, in first position', () => {
    const once = defaultTrainingCategorical(['camera', 'split'], ['camera'], ['split', 'loss_total'])
    expect(once).toEqual(['split', 'camera'])
    expect(defaultTrainingCategorical(['camera', 'split'], once, ['split'])).toEqual(once)
  })
  it('does nothing without a split from the run', () => {
    expect(defaultTrainingCategorical(['camera', 'split'], ['camera'], ['loss_total'])).toEqual(['camera'])
    expect(defaultTrainingCategorical(['camera'], ['camera'], ['split'])).toEqual(['camera'])
  })
})
