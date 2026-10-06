import { Loader2 } from 'lucide-react'
import type { RunSummary } from '../api'
import type { Comparison, MatchInfo, Overlay } from '../hooks/useRunOverlay'

interface Props {
  runs: RunSummary[]
  overlay: Overlay | null
  compare: Comparison | null
  match: MatchInfo | null
  loading: boolean
  error: string | null
  onSelect: (runId: string) => void
  onClear: () => void
}

/** "Run" dropdown for the tab bar: pick a training run whose per-sample losses become dataset columns. */
export function RunOverlayPicker({ runs, overlay, compare, match, loading, error, onSelect, onClear }: Props) {
  const partial = match && match.matched < match.total
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 10 }}>
      <span style={{ fontSize: 11, color: 'var(--c-t3)' }}>Run</span>
      <select value={compare ? '__compare__' : (overlay?.runId ?? '')} aria-label="Training run overlay" disabled={loading}
              onChange={e => (e.target.value ? onSelect(e.target.value) : onClear())}
              style={{ fontSize: 11, borderRadius: 7, padding: '3px 8px', outline: 'none', maxWidth: 220,
                       background: 'var(--c-surface)', border: '1px solid var(--c-border)', color: 'var(--c-t1)' }}>
        <option value="">None</option>
        {compare && <option value="__compare__" disabled>Comparison of two runs (cmp_ columns)</option>}
        {runs.map(r => (
          <option key={r.run_id} value={r.run_id}>
            {(r.name || r.run_id.slice(0, 8))} · {r.status.toLowerCase()} · {r.magnidata['magnidata.scored_epochs']?.split(',').length ?? 0} epochs
          </option>
        ))}
      </select>
      {loading && <Loader2 size={12} style={{ animation: 'spin 0.7s linear infinite' }} />}
      {partial && (
        <span role="status" style={{ fontSize: 11, color: 'var(--c-t3)' }}>
          {match.matched.toLocaleString()} / {match.total.toLocaleString()} samples have training data
        </span>
      )}
      {error && <span role="alert" title={error} style={{ fontSize: 11, color: 'var(--color-danger)' }}>Could not load run</span>}
    </div>
  )
}
