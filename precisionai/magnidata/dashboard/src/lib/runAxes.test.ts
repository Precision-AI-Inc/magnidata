import { describe, expect, it } from 'vitest'
import type { RunSummary } from '../api'
import { autoOverlayTarget, defaultTrainingAxes, highLossBrush, pickerEntries, pickerLabel } from './runAxes'

function run(over: Partial<RunSummary>): RunSummary {
  return {
    run_id: 'r', name: 'n', experiment_id: '1', status: 'FINISHED', start_time: 0, end_time: 0, duration_ms: 0,
    user: 'u', metrics: {}, params: {}, magnidata: {}, has_per_sample: false, has_split: false, mlflow_ui_url: null, ...over,
  }
}

describe('defaultTrainingAxes', () => {
  const runColumns = ['split', 'loss_total', 'loss_total_mean', 'loss_dice', 'hardness']
  const available = ['width', 'niqe', ...runColumns]
  it('appends the headline loss axes after the photometric ones', () => {
    expect(defaultTrainingAxes(available, ['width', 'niqe'], runColumns)).toEqual(['width', 'niqe', 'loss_total', 'loss_total_mean', 'hardness'])
  })
  it('adds nothing twice', () => {
    const once = defaultTrainingAxes(available, ['width'], runColumns)
    expect(defaultTrainingAxes(available, once, runColumns)).toEqual(once)
  })
  it('skips columns that are missing or did not come from the run', () => {
    expect(defaultTrainingAxes(['width', 'hardness'], ['width'], [])).toEqual(['width'])
    expect(defaultTrainingAxes(['width', 'loss_total'], ['width'], ['loss_total'])).toEqual(['width', 'loss_total'])
  })
  it('keeps the user\'s own axes untouched, even ones not in the list', () => {
    expect(defaultTrainingAxes(available, ['niqe', 'gone'], runColumns)[1]).toBe('gone')
  })
})

describe('highLossBrush', () => {
  const rows = [{ hardness: 0.1, loss_total_mean: 1 }, { hardness: 0.9, loss_total_mean: 5 }, { hardness: 0.7, loss_total_mean: 3 }]
  it('brushes hardness from 0.6 up to the observed maximum', () => {
    expect(highLossBrush(['hardness', 'loss_total_mean'], rows)).toEqual({ col: 'hardness', range: [0.6, 0.9] })
  })
  it('falls back to the top fifth of mean loss without hardness', () => {
    const many = Array.from({ length: 10 }, (_, i) => ({ loss_total_mean: i + 1 }))
    expect(highLossBrush(['loss_total_mean'], many)).toEqual({ col: 'loss_total_mean', range: [9, 10] })
  })
  it('returns null when nothing usable exists', () => {
    expect(highLossBrush(['hardness'], [{ hardness: null }])).toBeNull()
    expect(highLossBrush([], rows)).toBeNull()
  })
})

describe('pickerEntries', () => {
  it('lists every linked run, newest first, with a reason for unpickable ones', () => {
    const entries = pickerEntries([
      run({ run_id: 'old', start_time: 10, has_per_sample: true, has_split: true }),
      run({ run_id: 'queued', start_time: undefined, status: 'SCHEDULED' }),
      run({ run_id: 'live', start_time: 30, status: 'RUNNING' }),
      run({ run_id: 'split', start_time: 20, status: 'RUNNING', has_split: true }),
      run({ run_id: 'bad', start_time: 5, status: 'FAILED' }),
    ])
    expect(entries.map(e => e.run.run_id)).toEqual(['queued', 'live', 'split', 'old', 'bad'])
    const byId = Object.fromEntries(entries.map(e => [e.run.run_id, e]))
    expect(byId.queued).toMatchObject({ selectable: false, reason: 'queued' })
    expect(byId.live).toMatchObject({ selectable: false, reason: 'training · first losses after epoch 1 scoring' })
    expect(byId.split).toMatchObject({ selectable: true, reason: 'split only' })
    expect(byId.old).toMatchObject({ selectable: true, reason: null })
    expect(byId.bad).toMatchObject({ selectable: false, reason: 'failed — no per-sample data' })
  })
  it('labels status, device and scored epochs', () => {
    const [e] = pickerEntries([run({
      name: 'try', status: 'RUNNING', has_per_sample: true, params: { epochs: '10' },
      magnidata: { 'magnidata.scored_epochs': '1,2', 'magnidata.device': 'cuda:0 (x)' },
    })])
    expect(pickerLabel(e)).toBe('try · running · GPU · scored 2/10')
  })
})

describe('autoOverlayTarget', () => {
  const entries = pickerEntries([
    run({ run_id: 'a', status: 'RUNNING', start_time: 2, has_split: true }),
    run({ run_id: 'b', status: 'RUNNING', start_time: 1 }),
  ])
  it('prefers the launched run once it is pickable', () => {
    expect(autoOverlayTarget(entries, 'a', new Set())?.run.run_id).toBe('a')
    expect(autoOverlayTarget(entries, 'b', new Set())).toBeNull()
  })
  it('falls back to the newest pickable running run and never repeats', () => {
    expect(autoOverlayTarget(entries, null, new Set())?.run.run_id).toBe('a')
    expect(autoOverlayTarget(entries, null, new Set(['a']))).toBeNull()
  })
})
