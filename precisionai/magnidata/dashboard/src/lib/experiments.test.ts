import { describe, expect, it } from 'vitest'
import type { RunSummary } from '../api'
import {
  buildFilter, buildOrderBy, defaultSelection, deviceBadge, deviceLabel, formatDuration, formatEta, formatMetric, metricColumns, statusTone,
} from './experiments'

function run(over: Partial<RunSummary>): RunSummary {
  return {
    run_id: 'r', name: 'n', experiment_id: '1', status: 'FINISHED', start_time: 0, end_time: 0, duration_ms: 0,
    user: 'u', metrics: {}, params: {}, magnidata: {}, has_per_sample: false, has_split: false, mlflow_ui_url: null, ...over,
  }
}

describe('metricColumns', () => {
  it('unions keys across runs with different keys, sorted', () => {
    const cols = metricColumns([
      run({ metrics: { 'val_loss/loss': 1, 'val_metric/iou_1': 0.2 }, params: { lr: '0.1' } }),
      run({ metrics: { 'train_loss/loss': 2, 'val_loss/loss': 3 }, params: { epochs: '5', lr: '0.2' } }),
    ])
    expect(cols.metrics).toEqual(['train_loss/loss', 'val_loss/loss', 'val_metric/iou_1'])
    expect(cols.params).toEqual(['epochs', 'lr'])
  })
  it('handles no runs', () => {
    expect(metricColumns([])).toEqual({ metrics: [], params: [] })
  })
})

describe('formatDuration', () => {
  it('formats seconds, minutes and hours', () => {
    const d = (duration_ms: number | null) => formatDuration({ start_time: 1, end_time: 2, duration_ms })
    expect(d(42_000)).toBe('42 s')
    expect(d(185_000)).toBe('3 min 05 s')
    expect(d(2 * 3600_000 + 7 * 60_000)).toBe('2 h 07 min')
  })
  it('counts up for running runs and shows a dash without data', () => {
    expect(formatDuration({ start_time: 10_000, end_time: null, duration_ms: null }, 75_000)).toBe('1 min 05 s')
    expect(formatDuration({ start_time: null as unknown as number, end_time: null, duration_ms: null })).toBe('—')
  })
})

describe('statusTone', () => {
  it('maps statuses', () => {
    expect(statusTone('FINISHED')).toBe('success')
    expect(statusTone('FAILED')).toBe('danger')
    expect(statusTone('RUNNING')).toBe('info')
    expect(statusTone('SCHEDULED')).toBe('warning')
    expect(statusTone('KILLED')).toBe('neutral')
  })
})

describe('buildFilter', () => {
  it('is empty without constraints', () => {
    expect(buildFilter({})).toBe('')
  })
  it('combines status, dataset and clause text with AND', () => {
    expect(buildFilter({ status: 'FAILED', dataset: 'data_user/a.csv', text: "metrics.`val_loss/loss` < 0.5" })).toBe(
      "attributes.status = 'FAILED' AND tags.`magnidata.dataset_source` = 'data_user/a.csv' AND metrics.`val_loss/loss` < 0.5",
    )
  })
  it('turns plain words into a run-name search and strips quotes', () => {
    expect(buildFilter({ text: " my 'run' " })).toBe("attributes.run_name LIKE '%my run%'")
    expect(buildFilter({ dataset: "a'b.csv" })).toBe("tags.`magnidata.dataset_source` = 'ab.csv'")
  })
})

describe('buildOrderBy', () => {
  it('maps columns', () => {
    expect(buildOrderBy('metric', 'val_loss/loss', 'asc')).toBe('metrics.`val_loss/loss` ASC')
    expect(buildOrderBy('param', 'lr', 'desc')).toBe('params.`lr` DESC')
    expect(buildOrderBy('attribute', 'start', 'desc')).toBe('attributes.start_time DESC')
    expect(buildOrderBy('attribute', 'user', 'desc')).toBeNull()
  })
})

describe('defaultSelection', () => {
  it('keeps magnidata experiments and the most recent others', () => {
    const exps = [
      { experiment_id: '1', name: 'magnidata/a', last_update_time: 1 },
      { experiment_id: '2', name: 'x', last_update_time: 5 },
      { experiment_id: '3', name: 'y', last_update_time: 4 },
      { experiment_id: '4', name: 'z', last_update_time: 3 },
      { experiment_id: '5', name: 'w', last_update_time: 2 },
    ]
    expect(defaultSelection(exps).sort()).toEqual(['1', '2', '3', '4'])
  })
})

describe('formatMetric', () => {
  it('formats numbers compactly', () => {
    expect(formatMetric(undefined)).toBe('—')
    expect(formatMetric(12)).toBe('12')
    expect(formatMetric(0.123456)).toBe('0.1235')
    expect(formatMetric(0.00001234)).toBe('1.23e-5')
  })
})

describe('device and ETA', () => {
  it('maps the device tag to a badge', () => {
    expect(deviceBadge('cuda:0 (NVIDIA GeForce RTX 5070 Laptop GPU)')).toBe('GPU')
    expect(deviceBadge('cpu')).toBe('CPU')
    expect(deviceBadge(undefined)).toBeNull()
  })
  it('formats the ETA only for running runs', () => {
    expect(formatEta('RUNNING', '42')).toBe('ETA 42 s')
    expect(formatEta('RUNNING', '660')).toBe('ETA 11 min')
    expect(formatEta('RUNNING', '7500')).toBe('ETA 2 h 05 min')
    expect(formatEta('FINISHED', '660')).toBeNull()
    expect(formatEta('RUNNING', undefined)).toBeNull()
    expect(formatEta('RUNNING', 'abc')).toBeNull()
  })
  it('combines badge and ETA', () => {
    const magnidata = { 'magnidata.device': 'cuda:0 (x)', 'magnidata.eta_seconds': '660' }
    expect(deviceLabel({ status: 'RUNNING', magnidata })).toBe('GPU · ETA 11 min')
    expect(deviceLabel({ status: 'FINISHED', magnidata })).toBe('GPU')
    expect(deviceLabel({ status: 'FINISHED', magnidata: {} })).toBe('—')
  })
})
