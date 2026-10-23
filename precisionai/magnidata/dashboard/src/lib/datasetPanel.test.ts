import { describe, expect, it } from 'vitest'
import { formatBytes, formatNumber, histogramHeights, lineageLinks, maskCoverage } from './datasetPanel'

describe('dataset panel helpers', () => {
  it('formats byte sizes', () => {
    expect(formatBytes(undefined)).toBe('—')
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2_500)).toBe('3 KB')
    expect(formatBytes(3_400_000)).toBe('3.4 MB')
    expect(formatBytes(2_000_000_000)).toBe('2.0 GB')
  })

  it('reports mask coverage as an estimate from the sample', () => {
    expect(maskCoverage({ mask_coverage: { checked: 0, with_masks: 0 } }).fraction).toBeNull()
    const half = maskCoverage({ mask_coverage: { checked: 60, with_masks: 30 } })
    expect(half.fraction).toBe(0.5)
    expect(half.label).toBe('50% of 60 sampled images have masks')
  })

  it('scales histogram bins to the tallest bar', () => {
    expect(histogramHeights([1, 4, 2])).toEqual([0.25, 1, 0.5])
    expect(histogramHeights([0, 0])).toEqual([0, 0])
  })

  it('formats numbers compactly', () => {
    expect(formatNumber(12000)).toBe('12,000')
    expect(formatNumber(0.123456)).toBe('0.123')
    expect(formatNumber(Number.NaN)).toBe('—')
  })

  it('finds parents and children of a dataset', () => {
    const cloud = { name: 'Cloud', description: '', source: 'cloud' }
    const mini = { name: 'Mini', description: '', source: 'mini', parent_sources: ['cloud'] }
    const links = lineageLinks(mini, [cloud, mini])
    expect(links.parents).toEqual([cloud])
    expect(links.children).toEqual([])
    expect(lineageLinks(cloud, [cloud, mini]).children).toEqual([mini])
  })
})
