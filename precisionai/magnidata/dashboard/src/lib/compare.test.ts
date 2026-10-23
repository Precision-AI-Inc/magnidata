import { describe, expect, it } from 'vitest'
import { assignRunColors, bestValue, compareHash, defaultDirection, diffParams, parseCompareHash } from './compare'

describe('diffParams', () => {
  const runs: { params: Record<string, string> }[] = [
    { params: { lr: '0.1', epochs: '5', only_a: 'x' } },
    { params: { lr: '0.2', epochs: '5' } },
  ]
  it('hides params with equal values and counts missing keys as differing', () => {
    const rows = diffParams(runs)
    expect(rows.map(r => r.key)).toEqual(['lr', 'only_a'])
    expect(rows[0].values).toEqual(['0.1', '0.2'])
    expect(rows[1].values).toEqual(['x', undefined])
  })
  it('can return every param, flagging differences', () => {
    const rows = diffParams(runs, false)
    expect(rows.map(r => [r.key, r.differs])).toEqual([['epochs', false], ['lr', true], ['only_a', true]])
  })
  it('handles no runs', () => {
    expect(diffParams([])).toEqual([])
  })
})

describe('bestValue', () => {
  it('picks the minimum or maximum and ignores non-finite values', () => {
    expect(bestValue([3, 1, 2], 'min')).toBe(1)
    expect(bestValue([3, 1, 2], 'max')).toBe(3)
    expect(bestValue([NaN, 2], 'min')).toBe(2)
    expect(bestValue([], 'min')).toBeUndefined()
  })
})

describe('defaultDirection', () => {
  it('minimises losses and maximises IoU-like metrics', () => {
    expect(defaultDirection('val_loss/dice')).toBe('min')
    expect(defaultDirection('val_metric/miou')).toBe('max')
    expect(defaultDirection('val_metric/iou_background')).toBe('max')
    expect(defaultDirection('custom/acc')).toBe('max')
    expect(defaultDirection('something_else')).toBe('min')
  })
})

describe('assignRunColors', () => {
  it('is stable for the same ids and distinct within ten runs', () => {
    const ids = Array.from({ length: 10 }, (_, i) => `run${i}`)
    expect(assignRunColors(ids)).toEqual(assignRunColors(ids))
    expect(new Set(Object.values(assignRunColors(ids))).size).toBe(10)
    expect(assignRunColors(['b', 'a']).b).toBe(assignRunColors(['b']).b)
  })
})

describe('compare hash', () => {
  const a = 'a'.repeat(32)
  const b = 'b'.repeat(32)
  it('round-trips and rejects junk', () => {
    expect(parseCompareHash(compareHash([a, b]))).toEqual([a, b])
    expect(parseCompareHash('#other')).toEqual([])
    expect(parseCompareHash(`#compare=${a},nope,${b}`)).toEqual([a, b])
  })
})
