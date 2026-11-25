import type { CSVData, CSVRow } from '../types'
import { imageNameOf } from '../hooks/usePartitions'

/**
 * Title for a row's image preview: its stem column when the dataset has one, else the
 * image's file name, else (no image column either) the 1-based row number. The zero-based
 * `_idx` alone reads as a meaningless "0" above "Image 1 of N".
 */
export function previewTitle(row: CSVRow, data: Pick<CSVData, 'stemCol' | 'imagePath'>): string {
  const stem = data.stemCol ? String(row[data.stemCol] ?? '') : ''
  if (stem) return stem
  const name = imageNameOf(row, data as CSVData)
  if (name) return name
  return `Row ${row._idx + 1}`
}
