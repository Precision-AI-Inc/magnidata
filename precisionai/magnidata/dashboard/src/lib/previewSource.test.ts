import { describe, expect, it } from 'vitest'
import { previewCaption, previewSource } from './previewSource'
import type { PreviewTab } from './previewSource'

const base = { imgPath: 'ds/images/a.png', baseName: 'a.png', alpha: 0.5, runId: 'r1' }

describe('previewSource', () => {
  it('serves ground truth for Mask and Overlay and the run prediction for Prediction', () => {
    expect(previewSource({ ...base, tab: 'mask', predictionEpoch: 3 })).toContain('/annotation/mask')
    expect(previewSource({ ...base, tab: 'overlay', predictionEpoch: 3 })).toContain('/annotation/overlay')
    const prediction = previewSource({ ...base, tab: 'prediction', predictionEpoch: 3 })
    expect(prediction).toContain('/runs/r1/prediction')
    expect(prediction).toContain('epoch=3')
    expect(prediction).not.toContain('annotation')
  })

  it('never shows the ground truth in place of a missing prediction', () => {
    expect(previewSource({ ...base, tab: 'prediction', predictionEpoch: null })).toBe('')
    expect(previewSource({ ...base, tab: 'prediction', runId: undefined, predictionEpoch: 3 })).toBe('')
  })

  it('uses a different URL for every tab', () => {
    const tabs: PreviewTab[] = ['image', 'mask', 'overlay', 'prediction']
    const urls = tabs.map(tab => previewSource({ ...base, tab, predictionEpoch: 1 }))
    expect(new Set(urls).size).toBe(tabs.length)
  })

  it('shows nothing without an image path', () => {
    expect(previewSource({ ...base, imgPath: '', tab: 'image', predictionEpoch: null })).toBe('')
  })
})

describe('previewCaption', () => {
  it('labels the source of each mask view', () => {
    expect(previewCaption('mask', null)).toBe('Ground truth · dataset annotation')
    expect(previewCaption('overlay', 2)).toBe('Ground truth · dataset annotation')
    expect(previewCaption('prediction', 2)).toBe('Model prediction · epoch 2')
    expect(previewCaption('prediction', null)).toBeNull()
    expect(previewCaption('image', null)).toBeNull()
  })
})
