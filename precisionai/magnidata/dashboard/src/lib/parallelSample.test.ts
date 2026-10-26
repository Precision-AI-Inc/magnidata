import { describe, expect, it } from 'vitest'
import { sampleParallelRows } from './parallelSample'

describe('sampleParallelRows', () => {
  it('returns small inputs unchanged', () => {
    const rows = [1, 2, 3]
    expect(sampleParallelRows(rows, 4)).toBe(rows)
  })

  it('selects a deterministic, evenly distributed sample with both endpoints', () => {
    const rows = Array.from({ length: 10 }, (_, index) => index)
    expect(sampleParallelRows(rows, 4)).toEqual([0, 3, 6, 9])
    expect(sampleParallelRows(rows, 4)).toEqual(sampleParallelRows(rows, 4))
  })

  it('supports a single-line rendering limit', () => {
    expect(sampleParallelRows([4, 5, 6], 1)).toEqual([4])
  })
})
