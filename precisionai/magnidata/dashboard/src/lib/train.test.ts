import { describe, expect, it } from 'vitest'
import { isActive, launchParams, paramsFromRun, progressLabel, trainDisabledReason, validateParams } from './train'
import type { Bound, TrainParams } from './train'

const bounds: Record<string, Bound> = {
  epochs: { type: 'int', min: 1, max: 200 },
  score_every: { type: 'int', min: 1, max: 'epochs' },
  batch_size: { type: 'int', min: 1, max: 64 },
  image_size: { type: 'int', min: 128, max: 1024, multiple_of: 32 },
  lr: { type: 'float', min: 1e-6, max: 0.1 },
  val_fraction: { type: 'float', min: 0.05, max: 0.5 },
  seed: { type: 'int', min: 0, max: 2 ** 31 - 1 },
  split_mode: { type: 'choice', choices: ['random', 'column'] },
  loss_terms: { type: 'list', choices: ['dice', 'cce', 'bce'], min_items: 1 },
  loss_weights: { type: 'map', min: 0, max: 10 },
  top_k_predictions: { type: 'int', min: 0, max: 200 },
}

const valid: TrainParams = {
  epochs: 10, score_every: 1, batch_size: 8, image_size: 128, lr: 0.001, val_fraction: 0.2, seed: 0,
  split_mode: 'random', loss_terms: ['dice', 'cce'], loss_weights: {}, top_k_predictions: 50, pretrained: false, amp: true,
}

describe('validateParams', () => {
  it('accepts valid params', () => {
    expect(validateParams(valid, bounds)).toEqual({})
  })
  it('enforces min and max', () => {
    expect(validateParams({ ...valid, epochs: 0 }, bounds).epochs).toMatch(/between 1 and 200/)
    expect(validateParams({ ...valid, batch_size: 65 }, bounds).batch_size).toMatch(/between 1 and 64/)
    expect(validateParams({ ...valid, lr: 0.5 }, bounds).lr).toBeDefined()
  })
  it('requires image size to be a multiple of 32', () => {
    expect(validateParams({ ...valid, image_size: 130 }, bounds).image_size).toMatch(/multiple of 32/)
    expect(validateParams({ ...valid, image_size: 100 }, bounds).image_size).toBeDefined()
  })
  it('keeps score_every within epochs', () => {
    expect(validateParams({ ...valid, epochs: 5, score_every: 6 }, bounds).score_every).toMatch(/between 1 and 5/)
    expect(validateParams({ ...valid, epochs: 5, score_every: 5 }, bounds).score_every).toBeUndefined()
  })
  it('rejects non-integers and NaN', () => {
    expect(validateParams({ ...valid, epochs: 2.5 }, bounds).epochs).toMatch(/whole number/)
    expect(validateParams({ ...valid, lr: NaN }, bounds).lr).toMatch(/number/)
  })
  it('needs at least one known loss term and sane weights', () => {
    expect(validateParams({ ...valid, loss_terms: [] }, bounds).loss_terms).toMatch(/at least one/)
    expect(validateParams({ ...valid, loss_terms: ['focal'] }, bounds).loss_terms).toMatch(/unknown/)
    expect(validateParams({ ...valid, loss_weights: { dice: 11 } }, bounds).loss_weights).toBeDefined()
    expect(validateParams({ ...valid, loss_weights: { bce: 99 } }, bounds).loss_weights).toBeUndefined()
  })
  it('rejects an unknown split mode', () => {
    expect(validateParams({ ...valid, split_mode: 'x' }, bounds).split_mode).toBeDefined()
  })
})

describe('isActive', () => {
  it('is true for scheduled and running only', () => {
    expect(isActive('SCHEDULED')).toBe(true)
    expect(isActive('RUNNING')).toBe(true)
    expect(isActive('FINISHED')).toBe(false)
    expect(isActive(null)).toBe(false)
  })
})

describe('trainDisabledReason', () => {
  const ok = { hasDataset: true, mlflowReachable: true, workerAvailable: true, activeRunId: null }
  it('is null when everything is ready', () => {
    expect(trainDisabledReason(ok)).toBeNull()
  })
  it('reports the first cause in precedence order', () => {
    const none = { hasDataset: false, mlflowReachable: false, workerAvailable: false, activeRunId: 'x' }
    expect(trainDisabledReason(none)).toMatch(/dataset/)
    expect(trainDisabledReason({ ...none, hasDataset: true })).toMatch(/MLflow/)
    expect(trainDisabledReason({ ...none, hasDataset: true, mlflowReachable: true, workerReason: 'last seen 90 s ago' })).toMatch(/last seen 90 s ago/)
    expect(trainDisabledReason({ ...none, hasDataset: true, mlflowReachable: true, workerAvailable: true })).toMatch(/Another training run/)
  })
})

describe('progressLabel', () => {
  it('shows queued, then the epoch being trained', () => {
    expect(progressLabel('SCHEDULED', 0, 20)).toBe('queued')
    expect(progressLabel('RUNNING', 2, 20)).toBe('epoch 3/20')
    expect(progressLabel('RUNNING', 20, 20)).toBe('epoch 20/20')
    expect(progressLabel('RUNNING', 0, 20, {
      'magnidata.progress_stage': 'Preparing data',
      'magnidata.progress_current': '250',
      'magnidata.progress_total': '1000',
    })).toBe('Preparing data · 25%')
    expect(progressLabel('RUNNING', 0, 20, {
      'magnidata.progress_stage': 'Epoch 1/20 validating',
      'magnidata.progress_current': '0',
      'magnidata.progress_total': '0',
    })).toBe('Epoch 1/20 validating')
  })
})


describe('loader and AMP parameters', () => {
  const withWorkers = { ...bounds, num_workers: { type: 'int', min: 0, max: 16 } as Bound }
  it('accepts an unset or in-range loader count and rejects others', () => {
    expect(validateParams({ ...valid, num_workers: undefined }, withWorkers)).toEqual({})
    expect(validateParams({ ...valid, num_workers: 4 }, withWorkers)).toEqual({})
    expect(validateParams({ ...valid, num_workers: 17 }, withWorkers).num_workers).toMatch(/between 0 and 16/)
    expect(validateParams({ ...valid, num_workers: 1.5 }, withWorkers).num_workers).toMatch(/whole/)
  })
  it('restores amp and num_workers from an earlier run', () => {
    expect(paramsFromRun({ amp: 'false', num_workers: '3' })).toEqual({ amp: false, num_workers: 3 })
  })
})

describe('paramsFromRun', () => {
  it('turns MLflow param strings back into typed settings', () => {
    const out = paramsFromRun({
      epochs: '12', lr: '0.002', split_mode: 'column', pretrained: 'true', loss_terms: 'dice,bce', weight_dice: '2.5', preset: 'tiny_unet',
    })
    expect(out).toEqual({
      epochs: 12, lr: 0.002, split_mode: 'column', pretrained: true, loss_terms: ['dice', 'bce'], loss_weights: { dice: 2.5 },
    })
  })
  it('skips missing and non-numeric values', () => {
    expect(paramsFromRun({ epochs: 'many' })).toEqual({})
    expect(paramsFromRun({})).toEqual({})
  })
  it('restores the AgriTune provider and preload, ignoring unknown values', () => {
    expect(paramsFromRun({ feature_provider: 'cached', feature_preload: 'device' }))
      .toEqual({ feature_provider: 'cached', feature_preload: 'device' })
    expect(paramsFromRun({ feature_provider: 'disk', feature_preload: 'ssd' })).toEqual({})
  })
})

describe('validateParams for AgriTune providers', () => {
  const agritune = {
    feature_provider: { type: 'choice' as const, choices: ['cached', 'hybrid', 'online'] },
    feature_preload: { type: 'choice' as const, choices: ['none', 'host', 'device'] },
  }

  it('accepts cached features with preload and no augmentation', () => {
    expect(validateParams({ ...valid, feature_provider: 'cached', feature_preload: 'host', augmentation_mode: 'none' }, agritune)).toEqual({})
  })
  it('rejects augmentation with cached features and unknown preload targets', () => {
    expect(validateParams({ ...valid, feature_provider: 'cached', augmentation_mode: 'online' }, agritune).augmentation_mode).toMatch(/hybrid/)
    expect(validateParams({ ...valid, feature_preload: 'disk' as TrainParams['feature_preload'] }, agritune).feature_preload).toMatch(/one of/)
  })
})

describe('launchParams', () => {
  const full = { ...valid, num_workers: 2, feature_provider: 'hybrid', agritune_loss: 'ce_dice' } as TrainParams

  it('drops the reference-only settings for AgriTune presets', () => {
    const sent = launchParams(full, 'agritune')
    for (const key of ['loss_terms', 'num_workers', 'split_mode', 'pretrained']) expect(sent).not.toHaveProperty(key)
    expect(sent.agritune_loss).toBe('ce_dice')
    expect(sent.epochs).toBe(full.epochs)
  })

  it('sends everything for reference presets', () => {
    expect(launchParams(full, 'reference')).toEqual(full)
  })

  it('sends preload only with cached features', () => {
    expect(launchParams({ ...full, feature_provider: 'cached', feature_preload: 'host' }, 'agritune').feature_preload).toBe('host')
    expect(launchParams({ ...full, feature_provider: 'hybrid', feature_preload: 'host' }, 'agritune')).not.toHaveProperty('feature_preload')
  })
})
