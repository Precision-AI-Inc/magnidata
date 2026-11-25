import { describe, expect, it } from 'vitest'
import { ApiError } from '../api'
import { lostBuildMessage, POLL_MS, pollRetryDelay, resumableJobId, skippedFilesTitle } from './buildPoll'

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
  it('explains that the API has no record of the build', () => {
    expect(lostBuildMessage(new ApiError('404: job not found', 404))).toMatch(/no longer knows this build/)
  })

  it('falls back to a generic message for other failures', () => {
    expect(lostBuildMessage(dropped)).toBe('Lost track of the build')
  })
})

describe('resumableJobId', () => {
  it('resumes a build the API reports as running', () => {
    expect(resumableJobId({ job_id: 'j1', status: 'staging', percent: 29, message: 'Downloading…' })).toBe('j1')
  })

  it('ignores no build, a finished build, and a job without an id', () => {
    expect(resumableJobId(null)).toBeNull()
    expect(resumableJobId({ job_id: 'j1', status: 'done', percent: 100, message: 'Done' })).toBeNull()
    expect(resumableJobId({ job_id: 'j1', status: 'error', percent: null, message: 'x' })).toBeNull()
    expect(resumableJobId({ status: 'staging', percent: 5, message: 'x' })).toBeNull()
  })
})

describe('skippedFilesTitle', () => {
  it('lists each unreadable file with its reason', () => {
    const job = {
      status: 'done' as const, percent: 100, message: 'Done — 2 unreadable files left out',
      skipped: [{ file: 'a/notes.png', reason: 'cannot identify image file' }, { file: 'cut.jpg', reason: 'image file is truncated' }],
    }
    expect(skippedFilesTitle(job)).toBe(
      'Left out because they could not be read as images:\na/notes.png: cannot identify image file\ncut.jpg: image file is truncated',
    )
  })

  it('has nothing to say when every file was read', () => {
    expect(skippedFilesTitle({ status: 'done', percent: 100, message: 'Done', skipped: [] })).toBeUndefined()
    expect(skippedFilesTitle({ status: 'done', percent: 100, message: 'Done' })).toBeUndefined()
  })

  it('caps a long list', () => {
    const skipped = Array.from({ length: 25 }, (_, i) => ({ file: `f${i}.png`, reason: 'bad' }))
    const title = skippedFilesTitle({ status: 'done', percent: 100, message: 'Done', skipped }) ?? ''
    expect(title.split('\n')).toHaveLength(22)
    expect(title.endsWith('… and 5 more')).toBe(true)
  })
})
