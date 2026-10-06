import { useEffect, useRef, useState } from 'react'
import { Pause, Play } from 'lucide-react'
import { nextEpoch, resolveEpoch } from '../lib/epochs'

interface Props {
  scoredEpochs: number[]
  /** Epoch shown now: a number, or 'latest'. */
  epoch: number | 'latest'
  loading: boolean
  onChange: (epoch: number | 'latest') => void
}

const PLAY_MS = 1500
const DEBOUNCE_MS = 200

/** Slider over a run's scored epochs, with a "latest" toggle and a play button that steps through them. */
export function EpochScrubber({ scoredEpochs, epoch, loading, onChange }: Props) {
  const current = resolveEpoch(scoredEpochs, epoch)
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null)
  const latest = useRef({ current, scoredEpochs, loading, onChange })
  latest.current = { current, scoredEpochs, loading, onChange }

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => {
      const s = latest.current
      if (s.loading || s.current == null) return
      const next = nextEpoch(s.scoredEpochs, s.current)
      if (next != null) s.onChange(next)
    }, PLAY_MS)
    return () => clearInterval(id)
  }, [playing])
  useEffect(() => () => { if (debounce.current) clearTimeout(debounce.current) }, [])

  if (scoredEpochs.length === 0 || current == null) return null
  const index = dragIndex ?? Math.max(0, scoredEpochs.indexOf(current))
  const shown = scoredEpochs[index] ?? current

  const slide = (i: number) => {
    setDragIndex(i)
    setPlaying(false)
    if (debounce.current) clearTimeout(debounce.current)
    debounce.current = setTimeout(() => { setDragIndex(null); onChange(scoredEpochs[i]) }, DEBOUNCE_MS)
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 10 }}>
      <button onClick={() => setPlaying(p => !p)} aria-label={playing ? 'Stop playing epochs' : 'Play through epochs'}
              title={playing ? 'Stop' : 'Play through the scored epochs'} className="btn-icon" style={{ width: 24, height: 24 }}>
        {playing ? <Pause size={12} /> : <Play size={12} />}
      </button>
      <input type="range" min={0} max={scoredEpochs.length - 1} step={1} value={index} aria-label="Epoch"
             disabled={scoredEpochs.length < 2} onChange={e => slide(Number(e.target.value))} style={{ width: 120 }} />
      <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: 'var(--c-t2)', minWidth: 54 }}>
        epoch {shown}
      </span>
      <label style={{ fontSize: 11, color: 'var(--c-t3)', display: 'flex', alignItems: 'center', gap: 4 }}>
        <input type="checkbox" checked={epoch === 'latest'} onChange={e => { setPlaying(false); onChange(e.target.checked ? 'latest' : current) }} />
        latest
      </label>
    </div>
  )
}
