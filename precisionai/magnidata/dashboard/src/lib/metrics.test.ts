import { describe, expect, it } from 'vitest'
import { batches, ema, groupMetricKeys, METRIC_KEYS_PER_REQUEST, pairTrainVal, toRelativeTime } from './metrics'

describe('groupMetricKeys', () => {
  it('groups by prefix in a fixed order and omits empty groups', () => {
    const groups = groupMetricKeys(['val_metric/miou', 'train_loss/loss', 'best_epoch', 'Learning_rate/lr', 'val_loss/dice', 'train_loss/dice'])
    expect(groups.map(g => g.group)).toEqual(['train_loss/', 'val_loss/', 'val_metric/', 'Learning_rate/', 'other'])
    expect(groups[0].keys).toEqual(['train_loss/dice', 'train_loss/loss'])
    expect(groupMetricKeys(['x'])).toEqual([{ group: 'other', keys: ['x'] }])
    expect(groupMetricKeys([])).toEqual([])
  })
})

describe('pairTrainVal', () => {
  it('pairs terms and keeps unpaired ones, total first', () => {
    const pairs = pairTrainVal(['val_loss/loss', 'train_loss/loss', 'train_loss/dice', 'val_loss/cce', 'val_metric/miou'])
    expect(pairs).toEqual([
      { term: 'loss', train: 'train_loss/loss', val: 'val_loss/loss' },
      { term: 'cce', val: 'val_loss/cce' },
      { term: 'dice', train: 'train_loss/dice' },
    ])
  })
})

describe('ema', () => {
  it('alpha 0 is the identity', () => {
    expect(ema([1, 5, 3], 0)).toEqual([1, 5, 3])
  })
  it('smooths a known 3-point series', () => {
    const out = ema([1, 3, 5], 0.5)
    expect(out[0]).toBe(1)
    expect(out[1]).toBeCloseTo(2)
    expect(out[2]).toBeCloseTo(3.5)
  })
  it('handles empty input', () => {
    expect(ema([], 0.9)).toEqual([])
  })
})

describe('toRelativeTime', () => {
  it('converts timestamps to seconds from the first point', () => {
    expect(toRelativeTime([
      { step: 0, value: 1, timestamp: 10_000 },
      { step: 1, value: 1, timestamp: 12_500 },
    ])).toEqual([0, 2.5])
    expect(toRelativeTime([])).toEqual([])
  })
})

describe('batches', () => {
  it('splits a long key list into requests of at most the API limit', () => {
    const keys = Array.from({ length: 142 }, (_, i) => `val_metric/iou_${i}`)
    const parts = batches(keys, METRIC_KEYS_PER_REQUEST)
    expect(parts).toHaveLength(8)
    expect(parts.every(part => part.length <= 20)).toBe(true)
    expect(parts.flat()).toEqual(keys)
  })
  it('returns no batches for no keys', () => {
    expect(batches([], 20)).toEqual([])
  })
})
