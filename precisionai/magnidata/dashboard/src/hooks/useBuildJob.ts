import { useCallback, useRef, useState } from 'react'
import { api } from '../api'
import type { BuildJobStatus, DatasetMeta } from '../api'

const POLL_MS = 1500

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

/** Tracks a dataset-build job at the App level so its progress stays visible (as a
 * badge in the Header) regardless of whether the dialog that started it is still open,
 * and survives navigating around the app while the build runs. */
export function useBuildJob(onDone: (dataset: DatasetMeta) => void) {
  const [job, setJob]     = useState<BuildJobStatus | null>(null)
  const [dismissed, setDismissed] = useState(false)
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const poll = useCallback((jobId: string) => {
    api.getBuildJob(jobId).then(status => {
      setJob(status)
      if (status.status === 'done') {
        if (status.dataset) onDone(status.dataset)
      } else if (status.status !== 'error') {
        pollTimer.current = setTimeout(() => poll(jobId), POLL_MS)
      }
    }).catch(e => {
      setJob({ status: 'error', percent: 0, message: 'Lost track of the build', error: errorMessage(e) })
    })
  }, [onDone])

  const start = useCallback((jobPromise: Promise<{ job_id: string }>) => {
    setDismissed(false)
    return jobPromise.then(({ job_id }) => {
      setJob({ status: 'queued', percent: 0, message: 'Queued…' })
      poll(job_id)
      return job_id
    }).catch(e => {
      setJob({ status: 'error', percent: 0, message: 'Could not start build', error: errorMessage(e) })
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
      setJob({ status: 'error', percent: 0, message: 'Upload failed', error: errorMessage(e) })
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

  const busy = !!job && job.status !== 'done' && job.status !== 'error'

  return { job: dismissed ? null : job, busy, startUpload, startDemo, startLocal, dismiss }
}
