import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { RunDetailData, TrainingPresets } from '../api'
import { buildFilter } from '../lib/experiments'
import { isActive } from '../lib/train'

const POLL_MS = 5000
const PRESETS_MS = 15000

export interface Toast { id: number; tone: 'success' | 'danger' | 'neutral'; message: string }

/**
 * Follows the UI-launched training run at App level so its progress stays visible from any
 * view: polls every 5 s while it is scheduled or running, reports how many epochs have
 * finished, and raises a toast when it ends. Also keeps the trainer/MLflow availability the
 * Train buttons depend on.
 */
export function useActiveRun() {
  const [run, setRun] = useState<RunDetailData | null>(null)
  const [epochsDone, setEpochsDone] = useState(0)
  const [presets, setPresets] = useState<TrainingPresets | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const trackedId = useRef<string | null>(null)
  const lastStatus = useRef<string | null>(null)
  const toastId = useRef(0)

  const toast = useCallback((tone: Toast['tone'], message: string) => {
    const id = ++toastId.current
    setToasts(prev => [...prev, { id, tone, message }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 7000)
  }, [])
  const dismissToast = useCallback((id: number) => setToasts(prev => prev.filter(t => t.id !== id)), [])

  const refresh = useCallback(async () => {
    const id = trackedId.current
    if (!id) return
    try {
      const next = await api.getRun(id)
      if (trackedId.current !== id) return
      setRun(next)
      if (isActive(next.status)) {
        const done = (await api.getRunMetrics(id, ['train_loss/loss'])).metrics['train_loss/loss']?.length ?? 0
        setEpochsDone(done)
      }
      const before = lastStatus.current
      lastStatus.current = next.status
      if (before && isActive(before) && !isActive(next.status)) {
        const name = next.name || id.slice(0, 8)
        if (next.status === 'FINISHED') toast('success', `Training finished: ${name}`)
        else if (next.status === 'FAILED') toast('danger', `Training failed: ${next.tags['magnidata.error'] ?? name}`)
        else toast('neutral', `Training cancelled: ${name}`)
      }
    } catch { /* transient: the next poll retries */ }
  }, [toast])

  const track = useCallback((runId: string) => {
    trackedId.current = runId
    lastStatus.current = null
    setRun(null)
    setEpochsDone(0)
    void refresh()
  }, [refresh])

  const loadPresets = useCallback(async () => {
    try { setPresets(await api.getTrainingPresets()) } catch { setPresets(null) }
  }, [])

  // Pick up a run that is already going (page reload, launched earlier).
  useEffect(() => {
    void (async () => {
      try {
        for (const status of ['RUNNING', 'SCHEDULED'] as const) {
          const filter = `${buildFilter({ status })} AND tags.\`magnidata.origin\` = 'ui'`
          const page = await api.searchRuns({ experimentIds: [], filter, maxResults: 1 })
          if (page.runs.length) { track(page.runs[0].run_id); return }
        }
      } catch { /* MLflow down: nothing to follow */ }
    })()
  }, [track])

  useEffect(() => {
    void loadPresets()
    const id = setInterval(() => void loadPresets(), PRESETS_MS)
    return () => clearInterval(id)
  }, [loadPresets])

  const active = isActive(run?.status)
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => void refresh(), POLL_MS)
    return () => clearInterval(id)
  }, [active, refresh])

  return { run, active, epochsDone, presets, loadPresets, track, toast, toasts, dismissToast, refresh }
}
