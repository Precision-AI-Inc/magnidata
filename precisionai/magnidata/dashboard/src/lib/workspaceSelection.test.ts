import { describe, expect, it } from 'vitest'
import { canMerge, nextSelection } from './workspaceSelection'

describe('nextSelection', () => {
  it('selects a single dataset and toggles it off when clicked again', () => {
    expect(nextSelection([], 'a', false)).toEqual(['a'])
    expect(nextSelection(['b'], 'a', false)).toEqual(['a'])
    expect(nextSelection(['a'], 'a', false)).toEqual([])
  })

  it('collapses a multi-selection on a plain click of one of its members', () => {
    expect(nextSelection(['a', 'b'], 'a', false)).toEqual(['a'])
  })

  it('adds and removes datasets with an additive click', () => {
    expect(nextSelection(['a'], 'b', true)).toEqual(['a', 'b'])
    expect(nextSelection(['a', 'b'], 'a', true)).toEqual(['b'])
  })
})

describe('canMerge', () => {
  it('needs two distinct datasets', () => {
    expect(canMerge([])).toBe(false)
    expect(canMerge(['a'])).toBe(false)
    expect(canMerge(['a', 'a'])).toBe(false)
    expect(canMerge(['a', 'b'])).toBe(true)
  })
})
