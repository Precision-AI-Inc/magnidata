import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError, setMlflowOffline } from './api'

const fetchMock = vi.fn()

beforeEach(() => {
  fetchMock.mockReset()
  fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => ({}), text: async () => '' })
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  setMlflowOffline(false)
  vi.unstubAllGlobals()
})

describe('a deployment with no MLflow server', () => {
  beforeEach(() => setMlflowOffline(true))

  it.each([
    ['status', () => api.getMlflowStatus()],
    ['presets', () => api.getTrainingPresets()],
    ['experiment counts', () => api.datasetExperimentCounts()],
    ['a run', () => api.getRun('abc')],
  ])('answers %s with a 503 without asking the server', async (_name, call) => {
    const failure = await call().catch((err: unknown) => err)
    expect(failure).toBeInstanceOf(ApiError)
    expect((failure as ApiError).status).toBe(503)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('still asks the server for everything else', async () => {
    await api.config()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/config')
  })
})

describe('a normal deployment', () => {
  it('asks the server for MLflow data', async () => {
    await api.getTrainingPresets()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/training/presets')
  })

  it('goes back to asking once the offline switch is turned off', async () => {
    setMlflowOffline(true)
    setMlflowOffline(false)
    await api.getMlflowStatus()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
