import { ApiError } from '../api'
import type { BuildJobStatus } from '../api'

export const POLL_MS = 1500
const MAX_RETRY_MS = 15_000
/** Total time a build-status poll may keep failing before the badge gives up (~3 minutes). */
const GIVE_UP_AFTER_MS = 180_000

/** Delay before retrying a failed build-status poll, or null to stop polling.
 *
 * A build keeps running on the server while the browser loses the connection, so a dropped
 * request (fetch's "Failed to fetch") or a 5xx from nginx while the API restarts is retried with
 * backoff. A 404 means the API has no record of the job (a build cut short by an API restart
 * is still known, and reported as an error), and retrying cannot bring it back; nor can any
 * other 4xx.
 */
export function pollRetryDelay(error: unknown, failures: number, failingForMs: number): number | null {
  if (error instanceof ApiError && error.status < 500) return null
  if (failingForMs >= GIVE_UP_AFTER_MS) return null
  return Math.min(POLL_MS * 2 ** Math.max(failures - 1, 0), MAX_RETRY_MS)
}

/** User-facing message for a build whose status can no longer be read. */
export function lostBuildMessage(error: unknown): string {
  return error instanceof ApiError && error.status === 404
    ? 'The API no longer knows this build; start it again'
    : 'Lost track of the build'
}

/** The job id to resume polling after a page load, when the API reports a build still running. */
export function resumableJobId(job: BuildJobStatus | null | undefined): string | null {
  if (!job?.job_id || job.status === 'done' || job.status === 'error') return null
  return job.job_id
}

/** Hover text listing the files a finished build left out because they could not be read. */
export function skippedFilesTitle(job: BuildJobStatus): string | undefined {
  const skipped = job.skipped ?? []
  if (skipped.length === 0) return undefined
  const lines = skipped.slice(0, 20).map(s => `${s.file}: ${s.reason}`)
  if (skipped.length > 20) lines.push(`… and ${skipped.length - 20} more`)
  return `Left out because they could not be read as images:\n${lines.join('\n')}`
}
