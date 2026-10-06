import { describe, expect, it } from 'vitest'
import { lineageLabel, parentColumnText } from './lineage'

describe('lineageLabel', () => {
  it('names the dataset and the parent run', () => {
    expect(lineageLabel({ datasetName: 'curated', parentRunId: 'abcdef0123456789', parentRunName: 'first' }))
      .toBe('Trained on curated, from run first')
  })
  it('falls back to a short id and a generic dataset', () => {
    expect(lineageLabel({ datasetName: null, parentRunId: 'abcdef0123456789' })).toBe('Trained on the dataset, from run abcdef01')
  })
  it('is null without a parent', () => {
    expect(lineageLabel({ datasetName: 'x', parentRunId: undefined })).toBeNull()
  })
})

describe('parentColumnText', () => {
  it('shows the parent name, a short id, or a dash', () => {
    expect(parentColumnText('abcdef0123', { abcdef0123: 'first' })).toBe('first')
    expect(parentColumnText('abcdef0123', {})).toBe('abcdef01')
    expect(parentColumnText(undefined, {})).toBe('—')
  })
})
