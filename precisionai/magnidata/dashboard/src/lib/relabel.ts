import type { CSVData, CSVRow } from '../types'

/** CSV-escape a cell: quote when it contains a comma, quote or newline. */
export function csvCell(value: unknown): string {
  if (value == null) return ''
  const text = typeof value === 'number' ? String(value) : String(value)
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

const DERIVED_SUFFIX = /_(mean|delta|slope|rank)$/

/** Loss terms present among the overlaid run's columns (`loss_dice` when `loss_dice_rank` exists), total excluded. */
export function lossTermColumns(runColumns: string[]): string[] {
  const set = new Set(runColumns)
  return runColumns.filter(c => c.startsWith('loss_') && c !== 'loss_total' && !DERIVED_SUFFIX.test(c) && set.has(`${c}_rank`))
}

const basename = (p: unknown) => String(p ?? '').split(/[\\/]/).pop() ?? ''

/**
 * The relabel queue: one row per selected image with its training losses and any note, hardest first.
 * Columns: image_name, image_path, split, loss_total, loss_<term>…, hardness, iou, note.
 *
 * @param notes notes by image basename (several notes are joined by the caller)
 */
export function buildRelabelCsv(rows: CSVRow[], data: CSVData, notes: Record<string, string> = {}): string {
  const runCols = data.runColumns ?? []
  const terms = lossTermColumns(runCols)
  const hasIou = runCols.includes('iou')
  const header = ['image_name', 'image_path', 'split', 'loss_total', ...terms, 'hardness', ...(hasIou ? ['iou'] : []), 'note']
  const sorted = [...rows].sort((a, b) => ((b.hardness as number) ?? -1) - ((a.hardness as number) ?? -1))
  const lines = sorted.map(row => {
    const name = basename(data.imagePath ? row[data.imagePath] : '')
    return [
      name, data.imagePath ? row[data.imagePath] : '', row.split, row.loss_total, ...terms.map(t => row[t]),
      row.hardness, ...(hasIou ? [row.iou] : []), notes[name] ?? '',
    ].map(csvCell).join(',')
  })
  return [header.join(','), ...lines].join('\n') + '\n'
}
