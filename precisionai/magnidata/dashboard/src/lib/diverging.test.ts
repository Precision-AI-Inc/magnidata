import { describe, expect, it } from 'vitest'
import { divergingRange, interpolateDiverging, isDivergingColumn } from './diverging'

describe('isDivergingColumn', () => {
  it('flags signed change columns only', () => {
    expect(isDivergingColumn('cmp_hardness_change')).toBe(true)
    expect(isDivergingColumn('loss_total_delta')).toBe(true)
    expect(isDivergingColumn('loss_total')).toBe(false)
    expect(isDivergingColumn('hardness')).toBe(false)
  })
})

describe('divergingRange', () => {
  it('is symmetric about zero', () => {
    const [lo, hi] = divergingRange([-0.2, 0.1, 0.5, -0.4, 0.05])
    expect(lo).toBe(-hi)
    expect(hi).toBeGreaterThan(0)
  })
  it('ignores NaN and falls back for empty input', () => {
    expect(divergingRange([NaN])).toEqual([-1, 1])
    expect(divergingRange([])).toEqual([-1, 1])
    expect(divergingRange([0, 0])[1]).toBeGreaterThan(0)
  })
})

describe('interpolateDiverging', () => {
  it('is blue below, neutral at the centre and orange above', () => {
    const low = interpolateDiverging(0)
    const mid = interpolateDiverging(0.5)
    const high = interpolateDiverging(1)
    expect(low[2]).toBeGreaterThan(low[0])
    expect(high[0]).toBeGreaterThan(high[2])
    expect(Math.abs(mid[0] - mid[2])).toBeLessThan(0.05)
    expect(interpolateDiverging(-5)).toEqual(low)
    expect(interpolateDiverging(5)).toEqual(high)
  })
})
