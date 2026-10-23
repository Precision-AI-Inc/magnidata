export const GUIDE_HASH_PREFIX = '#guide='

export interface GuideChapterRef {
  id: string
  title: string
}

/** The chapter id named by a `#guide=<id>` hash, or null when the hash is absent or unknown. */
export function parseGuideHash(hash: string, ids: readonly string[]): string | null {
  if (!hash.startsWith(GUIDE_HASH_PREFIX)) return null
  const id = decodeURIComponent(hash.slice(GUIDE_HASH_PREFIX.length))
  return ids.includes(id) ? id : null
}

export function guideHash(id: string): string {
  return `${GUIDE_HASH_PREFIX}${encodeURIComponent(id)}`
}

export function adjacentChapters<T extends GuideChapterRef>(chapters: readonly T[], id: string): { prev: T | null; next: T | null } {
  const index = chapters.findIndex(chapter => chapter.id === id)
  if (index < 0) return { prev: null, next: null }
  return { prev: chapters[index - 1] ?? null, next: chapters[index + 1] ?? null }
}

/** Where a numbered callout sits on a screenshot, as percentages of its width and height. */
export interface Callout {
  x: number
  y: number
  text: string
}
