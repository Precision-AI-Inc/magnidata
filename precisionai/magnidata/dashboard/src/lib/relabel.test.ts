import { describe, expect, it } from 'vitest'
import type { CSVData, CSVRow } from '../types'
import { buildRelabelCsv, csvCell, lossTermColumns } from './relabel'

const runColumns = ['split', 'loss_total', 'loss_total_rank', 'loss_dice', 'loss_dice_mean', 'loss_dice_rank', 'hardness', 'iou']

function data(): CSVData {
  return {
    rows: [], cols: [], numericCols: [], categoricalCols: [], stringCols: [], catEncodings: {},
    imagePath: 'image_path', maskPath: null, thumbnailPath: null, stemCol: null, batchCol: null, clusterCol: null,
    fileName: 'd.csv', runColumns,
  }
}
const row = (i: number, name: string, hardness: number, split = 'train'): CSVRow =>
  ({ _idx: i, image_path: `data/${name}`, split, loss_total: 0.5, loss_dice: 0.25, hardness, iou: 0.8 })

describe('csvCell', () => {
  it('escapes commas, quotes and newlines', () => {
    expect(csvCell('plain')).toBe('plain')
    expect(csvCell('a,b')).toBe('"a,b"')
    expect(csvCell('say "hi"')).toBe('"say ""hi"""')
    expect(csvCell('two\nlines')).toBe('"two\nlines"')
    expect(csvCell(null)).toBe('')
    expect(csvCell(0)).toBe('0')
  })
})

describe('lossTermColumns', () => {
  it('finds base terms and skips derived columns', () => {
    expect(lossTermColumns(runColumns)).toEqual(['loss_dice'])
  })
})

describe('buildRelabelCsv', () => {
  it('uses the contract column order and only the given rows, hardest first', () => {
    const csv = buildRelabelCsv([row(0, 'a.png', 0.2), row(2, 'c.png', 0.9)], data())
    const lines = csv.trim().split('\n')
    expect(lines[0]).toBe('image_name,image_path,split,loss_total,loss_dice,hardness,iou,note')
    expect(lines).toHaveLength(3)
    expect(lines[1].startsWith('c.png,data/c.png,train,0.5,0.25,0.9,0.8')).toBe(true)
  })
  it('adds notes by image name and escapes them', () => {
    const csv = buildRelabelCsv([row(0, 'a.png', 0.5)], data(), { 'a.png': 'wrong class, "weed"\nsee mask' })
    expect(csv.trim().endsWith('"wrong class, ""weed""\nsee mask"')).toBe(true)
  })
  it('leaves the note empty when there is none', () => {
    expect(buildRelabelCsv([row(0, 'a.png', 0.5)], data()).trim().endsWith(',0.8,')).toBe(true)
  })
})
