import { describe, expect, it } from 'vitest'
import type { CSVData, CSVRow } from '../types'
import { analyzeCol, buildCatEncodings } from './columns'
import { imageKey, mergeRunColumns, removeRunColumns } from './mergeRunColumns'

function dataset(paths: string[]): CSVData {
  const rows: CSVRow[] = paths.map((p, i) => ({ _idx: i, image_path: p, width: 100 + i }))
  const cols = [analyzeCol('image_path', rows), analyzeCol('width', rows)]
  return {
    rows, cols, numericCols: ['width'], categoricalCols: [], stringCols: ['image_path'],
    catEncodings: buildCatEncodings([], rows), imagePath: 'image_path', maskPath: null, thumbnailPath: null,
    stemCol: null, batchCol: null, clusterCol: null, fileName: 'd.csv',
  }
}

const runRows = (epochValue: number) => [
  { image_name: 'a.png', split: 'train', loss_total: epochValue, hardness: 0.5 },
  { image_name: 'b.png', split: 'val', loss_total: epochValue * 2, hardness: 0.1 },
]

describe('imageKey', () => {
  it('takes the basename of unix and windows paths', () => {
    expect(imageKey('data_user/x/images/a.png')).toBe('a.png')
    expect(imageKey('C:\\imgs\\a.png')).toBe('a.png')
    expect(imageKey(null)).toBe('')
  })
})

describe('mergeRunColumns', () => {
  const data = dataset(['data_user/x/images/a.png', 'data_user/x/images/b.png', 'data_user/x/images/c.png'])

  it('joins on basename, types columns and counts unmatched rows as null', () => {
    const r = mergeRunColumns(data, runRows(1), 'image_path')
    expect(r.matched).toBe(2)
    expect(r.unmatched).toBe(1)
    expect(r.added).toEqual(['split', 'loss_total', 'hardness'])
    expect(r.data.rows[0].loss_total).toBe(1)
    expect(r.data.rows[2].loss_total).toBeNull()
    expect(r.data.numericCols).toEqual(['width', 'loss_total', 'hardness'])
    expect(r.data.categoricalCols).toEqual(['split'])
    expect(r.data.catEncodings.split.sorted.sort()).toEqual(['', 'train', 'val'])
    expect(r.data.runColumns).toEqual(r.added)
    expect(data.rows[0].loss_total).toBeUndefined()
  })

  it('replaces columns on a re-merge instead of duplicating them', () => {
    const first = mergeRunColumns(data, runRows(1), 'image_path').data
    const second = mergeRunColumns(first, runRows(5), 'image_path')
    expect(second.data.rows[0].loss_total).toBe(5)
    expect(second.data.cols.filter(c => c.name === 'loss_total')).toHaveLength(1)
    expect(second.data.numericCols.filter(c => c === 'loss_total')).toHaveLength(1)
  })

  it('removing restores the original column lists', () => {
    const merged = mergeRunColumns(data, runRows(1), 'image_path').data
    const cleared = removeRunColumns(merged)
    expect(cleared.numericCols).toEqual(data.numericCols)
    expect(cleared.categoricalCols).toEqual(data.categoricalCols)
    expect(cleared.cols.map(c => c.name)).toEqual(data.cols.map(c => c.name))
    expect(cleared.rows[0]).toEqual(data.rows[0])
    expect(cleared.runColumns).toBeUndefined()
  })

  it('uses the prefix and tolerates a dataset without an image column or run rows', () => {
    expect(mergeRunColumns(data, runRows(1), 'image_path', 'run_').added[1]).toBe('run_loss_total')
    expect(mergeRunColumns(data, runRows(1), null).matched).toBe(0)
    expect(mergeRunColumns(data, [], 'image_path').added).toEqual([])
  })

  it('does not offer columns of a run that matched nothing', () => {
    const none = mergeRunColumns(data, [{ image_name: 'zzz.png', split: 'train', loss_total: 1 }], 'image_path')
    expect(none.matched).toBe(0)
    expect(none.data.numericCols).toEqual(['width'])
    expect(none.data.categoricalCols).toEqual([])
  })
})
