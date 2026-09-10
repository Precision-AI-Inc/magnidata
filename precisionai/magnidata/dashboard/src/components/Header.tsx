import { useState } from 'react'
import { Download, X, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react'
import type { CSVData, CSVRow } from '../types'
import type { BuildJobStatus } from '../api'
import { api } from '../api'
import { useTheme } from '../hooks/useTheme'

interface Props {
  data: CSVData | null
  rows: CSVRow[]                    // current selection (filters + 3D selection applied) to export
  filteredCount: number
  embeddingsAvailable?: boolean
  datasetSource?: string | null
  buildJob?: BuildJobStatus | null
  onDismissBuildJob?: () => void
  onGoHome: () => void
  onOpenMagniData: () => void
  onOpenByod: () => void
  onOpenHelp: () => void
}

// The backend's step messages ("Extracting features…") don't always name the dataset.
function buildJobLabel(job: BuildJobStatus): string {
  return job.name && !job.message.includes(job.name) ? `${job.name} · ${job.message}` : job.message
}

function downloadBlob(content: string, filename: string, type: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([content], { type }))
  a.download = filename
  a.click()
  URL.revokeObjectURL(a.href)
}

// Markup contract copied from the shared Precision AI app shell
// (https://embeddings.precision.ai/ui/tokens.css, `.pai-shell*` classes) so this
// header renders pixel-identical to the reference site's — same brand mark, product
// nav, and theme toggle (including its CSS-driven sun/moon icon swap).
export function Header({ data, rows, filteredCount, embeddingsAvailable, datasetSource, buildJob, onDismissBuildJob, onGoHome, onOpenMagniData, onOpenByod, onOpenHelp }: Props) {
  const { toggle } = useTheme()

  // Export the current selection (rows passing active filters/3D selection) as CSV, plus
  // a matching embeddings JSON (filtered to the same rows) when the dataset has embeddings.
  const [exporting, setExporting] = useState(false)
  const exportSelection = async () => {
    if (!data || !rows.length) return
    const baseName = (data.fileName || 'dataset').replace(/\.csv$/i, '')

    const cols = Object.keys(rows[0]).filter(k => k !== '_idx')
    const esc = (v: unknown) => {
      const s = v == null ? '' : String(v)
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
    }
    const csv = [cols.join(','), ...rows.map(r => cols.map(c => esc(r[c])).join(','))].join('\n')
    downloadBlob(csv, `${baseName}_selection.csv`, 'text/csv')

    if (embeddingsAvailable && datasetSource && data.imagePath) {
      setExporting(true)
      try {
        const { embeddings } = await api.getEmbeddings(datasetSource)
        const keys = new Set(rows.map(r => String(r[data.imagePath!] ?? '')).filter(Boolean))
        const selected: Record<string, number[]> = {}
        for (const k of keys) if (embeddings[k]) selected[k] = embeddings[k]
        downloadBlob(JSON.stringify({ embeddings: selected }), `${baseName}_selection_embeddings.json`, 'application/json')
      } catch {
        // CSV already downloaded; embeddings are a best-effort addition.
      } finally {
        setExporting(false)
      }
    }
  }

  const nav = (label: string, onClick: () => void) => (
    <a href="#" onClick={e => { e.preventDefault(); onClick() }}>{label}</a>
  )

  return (
    <header className="pai-shell" style={{ flexShrink: 0 }}>
      <a className="pai-shell-brand" href="#" aria-label="Precision AI home"
         onClick={e => { e.preventDefault(); onGoHome() }}>
        <img className="brand-logo" src="/logo.png" alt="Precision AI" style={{ height: 28, width: 'auto' }} />
      </a>

      <nav className="pai-shell-nav" aria-label="Products">
        {nav('Home', onGoHome)}
        {nav('MagniData', onOpenMagniData)}
        {nav('BYOD', onOpenByod)}
        {nav('Help', onOpenHelp)}
      </nav>

      {/* Stats */}
      {data && (
        <div className="flex items-center gap-4">
          <Stat label="dataset" value={data.fileName} mono />
          <Stat label="rows" value={`${filteredCount.toLocaleString()} / ${data.rows.length.toLocaleString()}`} />
          <Stat label="cols" value={String(data.cols.length)} />
          <Stat label="numeric" value={String(data.numericCols.length)} />
        </div>
      )}

      {/* Global dataset-build progress — stays visible across the whole app, even once the
          dialog that started it is closed, until dismissed or the build finishes/fails. */}
      {buildJob && (
        <div className="flex items-center gap-2" style={{
          padding: '4px 10px', borderRadius: 'var(--radius-sm)', fontSize: 11.5, fontWeight: 600,
          background: buildJob.status === 'error' ? 'var(--color-danger-bg)' : 'var(--color-success-bg)',
          border: `1px solid ${buildJob.status === 'error' ? 'var(--color-danger)' : 'var(--color-success)'}`,
          color: buildJob.status === 'error' ? 'var(--color-danger)' : 'var(--color-success-text)',
        }}>
          {buildJob.status === 'done'
            ? <CheckCircle2 size={13} />
            : buildJob.status === 'error'
              ? <AlertTriangle size={13} />
              : <Loader2 size={13} style={{ animation: 'spin 0.7s linear infinite' }} />}
          <span style={buildJob.status === 'error' ? { fontWeight: 700, fontSize: 12 } : undefined}>
            {buildJob.status === 'error' ? (buildJob.error || buildJob.message) : buildJobLabel(buildJob)}
          </span>
          {buildJob.status !== 'error' && buildJob.status !== 'done' && (
            <span style={{ opacity: 0.75 }}>{buildJob.percent}%</span>
          )}
          {(buildJob.status === 'done' || buildJob.status === 'error') && onDismissBuildJob && (
            <button onClick={onDismissBuildJob} aria-label="Dismiss"
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', display: 'flex', padding: 0 }}>
              <X size={12} />
            </button>
          )}
        </div>
      )}

      <div className="pai-shell-right">
        {/* Export the current selection */}
        {data && (
          <button onClick={exportSelection} disabled={exporting || !rows.length}
                  title={embeddingsAvailable
                    ? 'Export the selected rows as CSV, plus their embeddings as JSON'
                    : 'Export the selected rows as CSV'}
                  style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, fontWeight: 600,
                           padding: '4px 10px', borderRadius: 'var(--radius-sm)',
                           cursor: exporting || !rows.length ? 'not-allowed' : 'pointer',
                           opacity: exporting || !rows.length ? 0.6 : 1,
                           color: 'var(--color-text-muted)', background: 'transparent',
                           border: '1px solid var(--color-border-strong)' }}>
            {exporting ? <Loader2 size={13} style={{ animation: 'spin 0.7s linear infinite' }} /> : <Download size={13} />}
            Export Selection
          </button>
        )}

        {/* Light / dark mode — icon visibility is driven entirely by tokens.css's
            .pai-theme-icon--sun/--moon rules keyed off [data-theme], so no JS state here. */}
        <button type="button" className="theme-toggle" onClick={toggle} aria-label="Toggle dark mode">
          <svg className="pai-theme-icon--sun" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32l1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41m11.32-11.32l1.41-1.41" />
          </svg>
          <svg className="pai-theme-icon--moon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
          </svg>
        </button>
      </div>
    </header>
  )
}

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex flex-col leading-none">
      <span className="text-[10px] uppercase tracking-widest" style={{ color: 'var(--color-text-subtle)' }}>{label}</span>
      <span className={`text-xs mt-0.5 ${mono ? 'font-mono' : 'font-medium'}`}
            style={{ color: 'var(--color-text)' }}>
        {value}
      </span>
    </div>
  )
}
