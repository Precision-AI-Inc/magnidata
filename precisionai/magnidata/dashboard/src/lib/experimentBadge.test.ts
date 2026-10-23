import { describe, expect, it } from 'vitest'
import { anyActive, experimentBadge } from './experimentBadge'

describe('experimentBadge', () => {
  it('is gray with no experiments', () => {
    expect(experimentBadge(undefined)).toEqual({ state: 'idle', label: 'No experiments' })
    expect(experimentBadge({ total: 0, active: 0 }).state).toBe('idle')
  })

  it('is yellow while any run is active', () => {
    expect(experimentBadge({ total: 5, active: 2 })).toEqual({ state: 'active', label: '2 running · 5 experiments total' })
    expect(experimentBadge({ total: 1, active: 1 }).label).toBe('1 running · 1 experiment total')
  })

  it('is gray with a count when every run has finished', () => {
    expect(experimentBadge({ total: 3, active: 0 })).toEqual({ state: 'idle', label: '3 experiments, none running' })
    expect(experimentBadge({ total: 1, active: 0 }).label).toBe('1 experiment, none running')
  })
})

describe('anyActive', () => {
  it('detects a live run anywhere', () => {
    expect(anyActive({ a: { total: 1, active: 0 } })).toBe(false)
    expect(anyActive({ a: { total: 1, active: 0 }, b: { total: 2, active: 1 } })).toBe(true)
  })
})
