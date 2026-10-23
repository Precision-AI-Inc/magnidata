import { describe, expect, it } from 'vitest'
import { ApiError } from '../api'
import { lostBuildMessage, POLL_MS, pollRetryDelay } from './buildPoll'

const dropped = new TypeError('Failed to fetch')

describe('pollRetryDelay', () => {
  it('retries a dropped connection with capped exponential backoff', () => {
    expect(pollRetryDelay(dropped, 1, 0)).toBe(POLL_MS)
    expect(pollRetryDelay(dropped, 2, 1_500)).toBe(2 * POLL_MS)
    expect(pollRetryDelay(dropped, 3, 4_500)).toBe(4 * POLL_MS)
    expect(pollRetryDelay(dropped, 10, 60_000)).toBe(15_000)
  })

  it('retries server errors such as the 502 nginx returns while the API restarts', () => {
    expect(pollRetryDelay(new ApiError('502: Bad Gateway', 502), 1, 0)).toBe(POLL_MS)
  })

  it('stops at once when the API no longer knows the job', () => {
    expect(pollRetryDelay(new ApiError('404: job not found', 404), 1, 0)).toBeNull()
  })

  it('gives up after about three minutes of failures', () => {
    expect(pollRetryDelay(dropped, 20, 179_999)).not.toBeNull()
    expect(pollRetryDelay(dropped, 20, 180_000)).toBeNull()
  })
})

describe('lostBuildMessage', () => {
  it('explains that a restart lost the build', () => {
    expect(lostBuildMessage(new ApiError('404: job not found', 404))).toMatch(/API restarted/)
  })

  it('falls back to a generic message for other failures', () => {
    expect(lostBuildMessage(dropped)).toBe('Lost track of the build')
  })
})
