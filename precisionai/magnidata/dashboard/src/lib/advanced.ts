/** Advanced AgriTune training settings: the API serves their specs, the Train dialog renders and sends them. */

export interface AdvancedSetting {
  key: string
  target: string
  group: string
  label: string
  type: 'int' | 'float' | 'bool' | 'choice' | 'ints' | 'floats' | 'range' | 'size'
  default: unknown
  help: string
  minimum: number | null
  maximum: number | null
  choices: string[]
  presets: string[]
  zero_is_none: boolean
}

/** What the dialog keeps per changed setting: a choice or flag as is, anything typed as its text. */
export type AdvancedInput = string | boolean

const LIST_TYPES = new Set(['ints', 'floats', 'range', 'size'])

/** Settings offered for `preset` (decoder settings belong to one preset each). */
export function advancedFor(settings: AdvancedSetting[], preset: string): AdvancedSetting[] {
  return settings.filter(s => !s.presets.length || s.presets.includes(preset))
}

/** Settings grouped by section, in the order the sections first appear. */
export function groupSettings(settings: AdvancedSetting[]): { group: string; settings: AdvancedSetting[] }[] {
  const groups = new Map<string, AdvancedSetting[]>()
  for (const s of settings) groups.set(s.group, [...(groups.get(s.group) ?? []), s])
  return [...groups].map(([group, items]) => ({ group, settings: items }))
}

/** The text a field shows for a value: lists and pairs comma-separated, unset as empty. */
export function formatValue(setting: AdvancedSetting, value: unknown): string {
  if (value === null || value === undefined) return ''
  if (Array.isArray(value)) return value.join(', ')
  return String(value)
}

/** The JSON value the API expects for an input, or an error message. Lists and pairs: empty unsets them. */
export function parseInput(setting: AdvancedSetting, input: AdvancedInput): { value?: unknown; error?: string } {
  if (setting.type === 'bool') return { value: input === true || input === 'true' }
  if (setting.type === 'choice') {
    return setting.choices.includes(String(input)) ? { value: String(input) } : { error: `must be one of ${setting.choices.join(', ')}` }
  }
  const text = String(input).trim()
  if (LIST_TYPES.has(setting.type)) {
    if (!text) return { value: [] }
    const numbers = text.split(/[\s,]+/).filter(Boolean).map(Number)
    if (numbers.some(n => !Number.isFinite(n))) return { error: 'must be numbers separated by commas' }
    if ((setting.type === 'range' || setting.type === 'size') && numbers.length !== 2) return { error: 'must be two numbers' }
    for (const n of numbers) {
      const error = checkNumber(setting, n)
      if (error) return { error }
    }
    if (setting.type === 'range' && numbers[0] > numbers[1]) return { error: 'the first number must not exceed the second' }
    return { value: numbers }
  }
  if (!text) return { error: 'must be a number' }
  const n = Number(text)
  if (!Number.isFinite(n)) return { error: 'must be a number' }
  const error = checkNumber(setting, n)
  return error ? { error } : { value: n }
}

function checkNumber(setting: AdvancedSetting, n: number): string | undefined {
  const whole = setting.type === 'int' || setting.type === 'ints' || setting.type === 'size'
  if (whole && !Number.isInteger(n)) return 'must be a whole number'
  const lo = setting.minimum ?? -Infinity
  const hi = setting.maximum ?? Infinity
  if (n < lo || n > hi) return `must be between ${lo} and ${hi}`
  return undefined
}

/** Inputs for the advanced settings an earlier run set (its MLflow param strings), to preload a retrain. */
export function advancedFromRun(settings: AdvancedSetting[], params: Record<string, string>): Record<string, AdvancedInput> {
  const out: Record<string, AdvancedInput> = {}
  for (const s of settings) {
    const text = params[s.key]
    if (text === undefined) continue
    out[s.key] = s.type === 'bool' ? text === 'true' : LIST_TYPES.has(s.type) ? text.split(',').filter(Boolean).join(', ') : text
  }
  return out
}

/** The request params and per-field errors for the changed advanced inputs. */
export function advancedPayload(settings: AdvancedSetting[], inputs: Record<string, AdvancedInput>): {
  params: Record<string, unknown>; errors: Record<string, string>
} {
  const params: Record<string, unknown> = {}
  const errors: Record<string, string> = {}
  for (const s of settings) {
    if (!(s.key in inputs)) continue
    const { value, error } = parseInput(s, inputs[s.key])
    if (error) errors[s.key] = error
    else params[s.key] = value
  }
  return { params, errors }
}
