import { describe, expect, it } from 'vitest'
import type { CSVRow } from '../types'
import { previewTitle } from './previewTitle'

const row = (fields: Record<string, unknown>): CSVRow => ({ _idx: 0, ...fields }) as unknown as CSVRow

describe('previewTitle', () => {
  it('uses the stem column when there is one', () => {
    expect(previewTitle(row({ stem: 'leaf_01', image_path: 'a/b/leaf_01.png' }), { stemCol: 'stem', imagePath: 'image_path' })).toBe('leaf_01')
  })

  it('falls back to the image file name when there is no stem column', () => {
    expect(previewTitle(row({ image_path: '/data/set/img_007.jpg' }), { stemCol: null, imagePath: 'image_path' })).toBe('img_007.jpg')
    expect(previewTitle(row({ image_path: 'C:\\set\\img.png' }), { stemCol: null, imagePath: 'image_path' })).toBe('img.png')
  })

  it('falls back to the file name when the stem cell is empty', () => {
    expect(previewTitle(row({ stem: '', image_path: 'x/y.png' }), { stemCol: 'stem', imagePath: 'image_path' })).toBe('y.png')
  })

  it('uses a 1-based row number when there is neither', () => {
    expect(previewTitle(row({}), { stemCol: null, imagePath: null })).toBe('Row 1')
  })
})
