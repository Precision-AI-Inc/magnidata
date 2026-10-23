import { describe, expect, it } from 'vitest'
import { nearestEarlierPrediction, nearestScoredEpoch, nextEpoch, resolveEpoch } from './epochs'

describe('nearestScoredEpoch', () => {
  it('snaps a slider value to a scored epoch', () => {
    expect(nearestScoredEpoch([2, 4, 8], 5)).toBe(4)
    expect(nearestScoredEpoch([2, 4, 8], 7)).toBe(8)
    expect(nearestScoredEpoch([2, 4, 8], 3)).toBe(2)
    expect(nearestScoredEpoch([2, 4, 8], 100)).toBe(8)
    expect(nearestScoredEpoch([], 1)).toBeNull()
  })
})

describe('nearestEarlierPrediction', () => {
  it('uses the exact epoch, else the closest earlier one, else none', () => {
    expect(nearestEarlierPrediction([1, 5, 9], 5)).toBe(5)
    expect(nearestEarlierPrediction([1, 5, 9], 7)).toBe(5)
    expect(nearestEarlierPrediction([5, 9], 3)).toBeNull()
    expect(nearestEarlierPrediction([], 3)).toBeNull()
  })
})

describe('nextEpoch', () => {
  it('advances and wraps', () => {
    expect(nextEpoch([1, 2, 5], 1)).toBe(2)
    expect(nextEpoch([1, 2, 5], 5)).toBe(1)
    expect(nextEpoch([], 1)).toBeNull()
  })
})

describe('resolveEpoch', () => {
  it('maps latest to the last scored epoch', () => {
    expect(resolveEpoch([1, 3], 'latest')).toBe(3)
    expect(resolveEpoch([1, 3], 2)).toBe(2)
    expect(resolveEpoch([], 'latest')).toBeNull()
  })
})
