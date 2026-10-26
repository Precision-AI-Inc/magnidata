import { CheckCircle2, ChevronRight, Home, AlertTriangle, Loader2, X } from 'lucide-react'
import type { CSVData } from '../types'
import type { BuildJobStatus } from '../api'
import { useTheme } from '../hooks/useTheme'

interface Props {
  pageTitle: string
  pageSubtitle: string
  data: CSVData | null
  filteredCount: number
  buildJob?: BuildJobStatus | null
  onDismissBuildJob?: () => void
  /** Live training run, shown like the build badge */
  trainingBadge?: string | null
}

// The backend's step messages ("Extracting features…") don't always name the dataset.
function buildJobLabel(job: BuildJobStatus): string {
  return job.name && !job.message.includes(job.name) ? `${job.name} · ${job.message}` : job.message
}

// Context bar for the MagniData application shell. Its theme toggle uses the
// same persisted token contract as the other Precision AI surfaces.
export function Header({ pageTitle, pageSubtitle, data, filteredCount, buildJob, onDismissBuildJob, trainingBadge }: Props) {
  const { toggle } = useTheme()

  return (
    <header className="dashboard-topbar">
      <div className="dashboard-topbar__title">
        <div className="dashboard-topbar__breadcrumb"><Home size={11} /><span>MagniData</span><ChevronRight size={11} /><strong>{pageTitle}</strong></div>
        <div className="dashboard-topbar__heading-row">
          <div><h1>{pageTitle}</h1><p>{pageSubtitle}</p></div>
          {data && (
            <div className="dashboard-topbar__context" title={`${data.cols.length} columns · ${data.numericCols.length} numeric`}>
              <span className="dashboard-topbar__dataset"><strong>{data.fileName}</strong><small>Active dataset</small></span>
              <span className="dashboard-topbar__count"><strong>{filteredCount.toLocaleString()}</strong><small>of {data.rows.length.toLocaleString()} samples</small></span>
            </div>
          )}
        </div>
      </div>

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

      {trainingBadge && (
        <div className="flex items-center gap-2" role="status" style={{
          padding: '4px 10px', borderRadius: 'var(--radius-sm)', fontSize: 11.5, fontWeight: 600,
          background: 'var(--color-success-bg)', border: '1px solid var(--color-success)', color: 'var(--color-success-text)',
        }}>
          <Loader2 size={13} style={{ animation: 'spin 0.7s linear infinite' }} />
          <span>{trainingBadge}</span>
        </div>
      )}

      <div className="dashboard-topbar__right">
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
