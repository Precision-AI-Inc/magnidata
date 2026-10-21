import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { CHAPTERS } from './chapters'
import { COMMANDS } from './commands'
import shots from './shots.json'

const here = path.dirname(new URL(import.meta.url).pathname)
const source = fs.readFileSync(path.join(here, 'chapters.tsx'), 'utf8')
const publicGuide = path.resolve(here, '../../../public/guide')

const used = (pattern: RegExp) => [...source.matchAll(pattern)].map(match => match[1])

describe('guide chapters', () => {
  it('has unique ids, titles and a body for every chapter', () => {
    const ids = CHAPTERS.map(chapter => chapter.id)
    expect(new Set(ids).size).toBe(ids.length)
    expect(CHAPTERS.length).toBeGreaterThanOrEqual(9)
    for (const chapter of CHAPTERS) {
      expect(chapter.title.length).toBeGreaterThan(3)
      expect(chapter.minutes).toBeGreaterThan(0)
      expect(typeof chapter.Body).toBe('function')
    }
  })
})

describe('guide screenshots', () => {
  const referenced = used(/<Shot name="([^"]+)"/g)

  it('only references screenshots that are declared', () => {
    for (const name of referenced) expect(Object.keys(shots)).toContain(name)
  })

  it('uses every declared screenshot', () => {
    expect(new Set(referenced)).toEqual(new Set(Object.keys(shots)))
  })

  it('has an image file, and alt text, for every screenshot', () => {
    for (const [name, meta] of Object.entries(shots)) {
      expect(fs.existsSync(path.join(publicGuide, `${name}.png`)), `${name}.png is missing; run scripts/capture-guide.mjs`).toBe(true)
      expect(meta.alt.length).toBeGreaterThan(10)
      expect(meta.caption.length).toBeGreaterThan(10)
    }
  })

  it('keeps every image below the 1,800 KB pre-commit limit', () => {
    for (const name of Object.keys(shots)) {
      expect(fs.statSync(path.join(publicGuide, `${name}.png`)).size).toBeLessThan(1_800_000)
    }
  })
})

describe('guide commands', () => {
  it('only shows commands from the shared list, and uses all of them', () => {
    const referenced = new Set(used(/<Cmd id="([^"]+)"/g))
    for (const id of referenced) expect(Object.keys(COMMANDS)).toContain(id)
    for (const id of Object.keys(COMMANDS)) expect(referenced, `command ${id} is never shown`).toContain(id)
  })

  describe('agrituneEnable', () => {
    // Run only the .env edit, not the rebuild, in a scratch directory.
    const enable = (env: string | null) => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'guide-env-'))
      if (env !== null) fs.writeFileSync(path.join(dir, '.env'), env)
      const script = COMMANDS.agrituneEnable.replace(/\ndocker compose .*$/, '')
      execFileSync('bash', ['-c', script], { cwd: dir })
      execFileSync('bash', ['-c', script], { cwd: dir })
      return fs.readFileSync(path.join(dir, '.env'), 'utf8')
    }

    it('creates COMPOSE_FILE when .env has none', () => {
      expect(enable(null)).toBe('COMPOSE_FILE=docker-compose.yml:docker-compose.agritune.yml\n')
    })

    it('extends an existing COMPOSE_FILE once, keeping the GPU file and other settings', () => {
      const env = enable('COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml\nMAGNIDATA_TRAIN_GPU=2\n')
      expect(env).toBe(
        'COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.agritune.yml\nMAGNIDATA_TRAIN_GPU=2\n',
      )
    })
  })

  it('runs repository tools from the repository root with module paths that exist', () => {
    const repo = path.resolve(here, '../../../../../..')
    for (const command of Object.values(COMMANDS)) {
      for (const match of command.matchAll(/python -m (precisionai[\w.]*)/g)) {
        const modulePath = path.join(repo, ...match[1].split('.')) + '.py'
        expect(fs.existsSync(modulePath), `${match[1]} does not exist`).toBe(true)
      }
    }
  })
})
