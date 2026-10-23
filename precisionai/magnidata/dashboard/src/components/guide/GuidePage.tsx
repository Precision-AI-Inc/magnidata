import { useCallback, useEffect, useState } from 'react'
import { ArrowLeft, ArrowRight, BookOpen, Clock } from 'lucide-react'
import { adjacentChapters, guideHash, parseGuideHash } from '../../lib/guide'
import { CHAPTERS } from './chapters'
import './GuidePage.css'

interface Props {
  onOpenByod?: () => void
}

const IDS = CHAPTERS.map(chapter => chapter.id)

export function GuidePage({ onOpenByod }: Props) {
  const [activeId, setActiveId] = useState<string>(() => parseGuideHash(window.location.hash, IDS) ?? IDS[0])
  const chapter = CHAPTERS.find(candidate => candidate.id === activeId) ?? CHAPTERS[0]
  const { prev, next } = adjacentChapters(CHAPTERS, chapter.id)
  const number = CHAPTERS.indexOf(chapter) + 1

  const open = useCallback((id: string) => {
    setActiveId(id)
    window.history.replaceState(null, '', guideHash(id))
    document.querySelector('.guide-page')?.scrollTo({ top: 0 })
  }, [])

  useEffect(() => {
    const onHash = () => {
      const id = parseGuideHash(window.location.hash, IDS)
      if (id) setActiveId(id)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  return (
    <div className="guide-page">
      <aside className="guide-toc" aria-label="Guide chapters">
        <div className="guide-toc__label"><BookOpen size={13} /> Guide</div>
        <nav>
          {CHAPTERS.map((entry, index) => (
            <button key={entry.id} data-active={entry.id === chapter.id} aria-current={entry.id === chapter.id ? 'page' : undefined}
                    onClick={() => open(entry.id)}>
              <span className="guide-toc__number">{index + 1}</span>
              <span>{entry.title}</span>
            </button>
          ))}
        </nav>
      </aside>

      <article className="guide-content" key={chapter.id}>
        <header className="guide-header">
          <span className="guide-header__eyebrow">Chapter {number} of {CHAPTERS.length}</span>
          <h1>{chapter.title}</h1>
          <p><Clock size={13} /> about {chapter.minutes} minutes · {chapter.summary}</p>
        </header>
        <chapter.Body />
        {chapter.id === 'agristress' && onOpenByod && (
          <button className="btn-primary guide-action" onClick={onOpenByod}>Open the Add dataset dialog</button>
        )}
        <nav className="guide-pager" aria-label="Chapter navigation">
          {prev ? <button onClick={() => open(prev.id)}><ArrowLeft size={15} /><span><small>Previous</small>{prev.title}</span></button> : <span />}
          {next ? <button onClick={() => open(next.id)}><span><small>Next</small>{next.title}</span><ArrowRight size={15} /></button> : <span />}
        </nav>
      </article>
    </div>
  )
}
