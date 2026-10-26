import { useCallback, useRef, useState } from 'react'
import { api, errorText } from '../api'
import type { BuildJobStatus, DatasetMeta } from '../api'
import { lostBuildMessage, POLL_MS, pollRetryDelay } from '../lib/buildPoll'

function isActive(job: BuildJobStatus | null): boolean {
  return !!job && job.status !== 'done' && job.status !== 'error'
}

/** A rejected start (e.g. 409 while another build runs) must not overwrite the running
 * job's badge — the poll loop would just flip it back, hiding the error and confusing
 * which build is actually in progress. */
function failedStart(message: string, e: unknown) {
  return (prev: BuildJobStatus | null): BuildJobStatus =>
    prev && isActive(prev) ? prev : { status: 'error', percent: 0, message, error: errorText(e) }
}

/** Tracks a dataset-build job at the App level so its progress stays visible (as a
 * badge in the Header) regardless of whether the dialog that started it is still open,
 * and survives navigating around the app while the build runs. */
export function useBuildJob(onDone: (dataset: DatasetMeta) => void) {
  const [job, setJob]     = useState<BuildJobStatus | null>(null)
  const [dismissed, setDismissed] = useState(false)
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const failures = useRef({ count: 0, since: 0 })

  const poll = useCallback((jobId: string) => {
    api.getBuildJob(jobId).then(status => {
      failures.current = { count: 0, since: 0 }
      setJob(status)
      if (status.status === 'done') {
        if (status.dataset) onDone(status.dataset)
      } else if (status.status !== 'error') {
        pollTimer.current = setTimeout(() => poll(jobId), POLL_MS)
      }
    }).catch(e => {
      const failed = failures.current
      failed.count += 1
      if (failed.count === 1) failed.since = Date.now()
      const delay = pollRetryDelay(e, failed.count, Date.now() - failed.since)
      if (delay === null) {
        setJob({ status: 'error', percent: 0, message: lostBuildMessage(e), error: errorText(e) })
        return
      }
      // Keep the last known progress; the build carries on server-side while we reconnect.
      setJob(prev => (prev ? { ...prev, message: 'Connection lost, retrying…' } : prev))
      pollTimer.current = setTimeout(() => poll(jobId), delay)
    })
  }, [onDone])

  const start = useCallback((jobPromise: Promise<{ job_id: string }>) => {
    setDismissed(false)
    return jobPromise.then(({ job_id }) => {
      setJob({ status: 'queued', percent: 0, message: 'Queued…' })
      poll(job_id)
      return job_id
    }).catch(e => {
      setJob(failedStart('Could not start build', e))
      throw e
    })
  }, [poll])

  const startUpload = useCallback((formData: FormData) => {
    setDismissed(false)
    setJob({ status: 'uploading', percent: 0, message: 'Uploading images…' })
    return api.buildDataset(formData, percent => {
      setJob({
        status: 'uploading',
        percent,
        message: percent >= 100 ? 'Finishing upload…' : 'Uploading images…',
      })
    }).then(({ job_id }) => {
      setJob({ status: 'queued', percent: 0, message: 'Queued…' })
      poll(job_id)
      return job_id
    }).catch(e => {
      setJob({ status: 'error', percent: 0, message: 'Upload failed', error: errorText(e) })
      throw e
    })
  }, [poll])
  const startDemo = useCallback(
    (demoKey: string) => start(api.buildDemo(demoKey)),
    [start],
  )
  const startLocal = useCallback(
    (folder: string, name?: string) => start(api.buildLocal(folder, name)),
    [start],
  )

  const dismiss = useCallback(() => {
    if (pollTimer.current) clearTimeout(pollTimer.current)
    setDismissed(true)
  }, [])

  const busy = isActive(job)

  return { job: dismissed ? null : job, busy, startUpload, startDemo, startLocal, dismiss }
}
