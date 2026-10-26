import { describe, expect, it } from 'vitest'
import { adjacentChapters, guideHash, parseGuideHash } from './guide'

const IDS = ['overview', 'install', 'partition']

describe('guide hash', () => {
  it('round-trips a chapter id', () => {
    expect(parseGuideHash(guideHash('partition'), IDS)).toBe('partition')
  })

  it('ignores other hashes and unknown chapters', () => {
    expect(parseGuideHash('#compare=a,b', IDS)).toBeNull()
    expect(parseGuideHash('#guide=nope', IDS)).toBeNull()
    expect(parseGuideHash('', IDS)).toBeNull()
  })
})

describe('adjacentChapters', () => {
  const chapters = IDS.map(id => ({ id, title: id }))

  it('returns the neighbours and null at the ends', () => {
    expect(adjacentChapters(chapters, 'install')).toEqual({ prev: chapters[0], next: chapters[2] })
    expect(adjacentChapters(chapters, 'overview').prev).toBeNull()
    expect(adjacentChapters(chapters, 'partition').next).toBeNull()
    expect(adjacentChapters(chapters, 'missing')).toEqual({ prev: null, next: null })
  })
})
