import { describe, expect, it } from 'vitest'
import { advancedFor, advancedFromRun, advancedPayload, formatValue, groupSettings, parseInput } from './advanced'
import type { AdvancedSetting } from './advanced'

const base = { target: '', help: '', minimum: null, maximum: null, choices: [], presets: [], zero_is_none: false }
const settings: AdvancedSetting[] = [
  { ...base, key: 'optimizer.name', group: 'Optimizer', label: 'Optimizer', type: 'choice', default: 'adamw', choices: ['adamw', 'adam', 'sgd'] },
  { ...base, key: 'optimizer.betas', group: 'Optimizer', label: 'Betas', type: 'range', default: [0.9, 0.999], minimum: 0, maximum: 0.9999 },
  { ...base, key: 'scheduler.warmup_steps', group: 'Learning-rate schedule', label: 'Warm-up', type: 'int', default: 0, minimum: 0, maximum: 1000 },
  { ...base, key: 'decoder.hidden_dims', group: 'Decoder', label: 'Hidden', type: 'ints', default: null, minimum: 1, maximum: 8192, presets: ['agritune_mlp_probe'] },
  { ...base, key: 'data.pin_memory', group: 'Data loading', label: 'Pin', type: 'bool', default: false },
]

describe('advancedFor and groupSettings', () => {
  it('keeps preset-specific settings to their preset and groups in order of appearance', () => {
    expect(advancedFor(settings, 'agritune_token_fpn').map(s => s.key)).not.toContain('decoder.hidden_dims')
    expect(advancedFor(settings, 'agritune_mlp_probe').map(s => s.key)).toContain('decoder.hidden_dims')
    expect(groupSettings(settings).map(g => g.group)).toEqual(['Optimizer', 'Learning-rate schedule', 'Decoder', 'Data loading'])
  })
})

describe('parseInput', () => {
  it('parses numbers, lists, pairs and flags into the values the API expects', () => {
    expect(parseInput(settings[2], ' 100 ')).toEqual({ value: 100 })
    expect(parseInput(settings[1], '0.8, 0.99')).toEqual({ value: [0.8, 0.99] })
    expect(parseInput(settings[3], '256 128')).toEqual({ value: [256, 128] })
    expect(parseInput(settings[3], '')).toEqual({ value: [] })
    expect(parseInput(settings[4], true)).toEqual({ value: true })
    expect(parseInput(settings[0], 'sgd')).toEqual({ value: 'sgd' })
  })
  it('explains what is wrong', () => {
    expect(parseInput(settings[2], '2.5').error).toMatch(/whole/)
    expect(parseInput(settings[2], '5000').error).toMatch(/between 0 and 1000/)
    expect(parseInput(settings[1], '0.9').error).toMatch(/two numbers/)
    expect(parseInput(settings[1], '0.99, 0.9').error).toMatch(/must not exceed/)
    expect(parseInput(settings[3], '64, x').error).toMatch(/numbers separated/)
    expect(parseInput(settings[0], 'lion').error).toMatch(/one of/)
  })
})

describe('advancedPayload and advancedFromRun', () => {
  it('sends only changed settings and collects errors by key', () => {
    const { params, errors } = advancedPayload(settings, { 'optimizer.name': 'sgd', 'scheduler.warmup_steps': '-1' })
    expect(params).toEqual({ 'optimizer.name': 'sgd' })
    expect(errors['scheduler.warmup_steps']).toMatch(/between/)
  })
  it('restores an earlier run for a retrain', () => {
    expect(advancedFromRun(settings, { 'optimizer.betas': '0.8,0.99', 'data.pin_memory': 'true', 'decoder.hidden_dims': '', lr: '0.001' }))
      .toEqual({ 'optimizer.betas': '0.8, 0.99', 'data.pin_memory': true, 'decoder.hidden_dims': '' })
  })
  it('formats defaults for placeholders', () => {
    expect(formatValue(settings[1], [0.9, 0.999])).toBe('0.9, 0.999')
    expect(formatValue(settings[3], null)).toBe('')
  })
})
