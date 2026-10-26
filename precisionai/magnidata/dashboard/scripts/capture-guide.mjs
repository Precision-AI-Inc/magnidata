#!/usr/bin/env node
/**
 * Captures the guide's screenshots from the running UI into public/guide/.
 *
 * Needs a stack whose data matches the guide: AgriStress-500 plus the two tiled sub-datasets made by
 * `partition_dataset --limit-images 8 --parts 2`, and (for the training stages) a trainer worker.
 * The `source` stage needs a second portal whose API has only AgriStress-500 registered, and the `byod`
 * stage a third whose API has no dataset at all (a fresh install).
 *
 *   node scripts/capture-guide.mjs --stage source,workspace,explorer
 *
 * Environment:
 *   GUIDE_URL         portal with the partitioned data       (default http://127.0.0.1:5174)
 *   GUIDE_SOURCE_URL  portal with only AgriStress-500        (default http://127.0.0.1:5176)
 *   GUIDE_EMPTY_URL   portal with no datasets (fresh install) (default http://127.0.0.1:5177)
 *   CHROME_PATH       Chrome/Chromium executable             (default: newest in ~/.cache/ms-playwright)
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright-core'

const here = path.dirname(fileURLToPath(import.meta.url))
const OUT = path.resolve(here, '../public/guide')
const URL = process.env.GUIDE_URL ?? 'http://127.0.0.1:5174'
const SOURCE_URL = process.env.GUIDE_SOURCE_URL ?? 'http://127.0.0.1:5176'
const EMPTY_URL = process.env.GUIDE_EMPTY_URL ?? 'http://127.0.0.1:5177'
const PART_1 = 'AgriStress-500 tiles part 1'
const PART_2 = 'AgriStress-500 tiles part 2'
const AGRITUNE_URL = process.env.GUIDE_AGRITUNE_URL ?? 'http://127.0.0.1:25175'
const AGRITUNE_TITLE = 'Field survey fixture'
const AGRITUNE_SOURCE = 'data_user/my-field-survey.csv'
// A tile the fixture run saved a prediction for (it is one of that run's hardest).
const AGRITUNE_TILE = 'pai-dpyusSdy_x02304_y00000.png'
const PART_1_SOURCE = 'data_user/agristress-500_tiles-part1.csv'
// A tile whose mask has three well-represented classes, so previews show something worth looking at.
const SAMPLE = 'pai-FdHBsOXP_x01280_y03584.png'

function findChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH
  const cache = path.join(os.homedir(), '.cache/ms-playwright')
  const builds = fs.existsSync(cache) ? fs.readdirSync(cache).filter(name => /^chromium-\d+$/.test(name)).sort() : []
  for (const build of builds.reverse()) {
    const candidate = path.join(cache, build, 'chrome-linux64/chrome')
    if (fs.existsSync(candidate)) return candidate
  }
  throw new Error('No Chrome found; set CHROME_PATH')
}

const wanted = (process.argv.find(arg => arg.startsWith('--stage=')) ?? process.argv[process.argv.indexOf('--stage') + 1] ?? '')
  .replace('--stage=', '').split(',').filter(Boolean)

const browser = await chromium.launch({ executablePath: findChrome(), headless: true, args: ['--no-sandbox'] })
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 })
const page = await context.newPage()
fs.mkdirSync(OUT, { recursive: true })

const settle = ms => page.waitForTimeout(ms)
const shot = async (name, target = page) => {
  await target.screenshot({ path: path.join(OUT, `${name}.png`) })
  console.log(`  captured ${name}`)
}
const node = title => page.locator('.react-flow__node').filter({ hasText: title }).first()
const home = async (url = URL) => {
  await page.goto(url)
  await page.evaluate(() => localStorage.clear())
  await page.goto(url)
  await page.locator('.react-flow__node').first().waitFor()
  await settle(900)
}
const openExplorer = async (title = PART_1) => {
  await home()
  await node(title).dblclick()
  await page.getByRole('heading', { name: 'Visual Explorer' }).waitFor()
  await page.locator('tbody tr').first().waitFor({ timeout: 60000 })
  await settle(2500)
}

/** Open the preview modal on the tile named `name` by stepping through the table with the arrow keys. */
const openPreview = async (name = SAMPLE, datasetSource = PART_1_SOURCE) => {
  const index = await page.evaluate(async ({ source, tile }) => {
    const text = await (await fetch(`/api/dataset/csv?source=${encodeURIComponent(source)}`)).text()
    return text.split('\n').slice(1).findIndex(line => line.includes(tile))
  }, { source: datasetSource, tile: name })
  if (index < 0) throw new Error(`tile ${name} not found in ${datasetSource}`)
  await page.locator('tbody tr button').first().click()
  await page.locator('.modal-box').waitFor()
  for (let step = 0; step < index; step += 1) await page.keyboard.press('ArrowRight')
  await settle(800)
}
const imagesLoaded = () => page.waitForFunction(
  () => [...document.querySelectorAll('.modal-box img')].every(img => img.complete && img.naturalWidth > 0),
)
const setupPanel = () => page.getByRole('complementary', { name: 'Visual Explorer configuration' })

/** Overlay the finished run, brush its persistently hard tiles and leave the explorer with them selected. */
const brushHardTiles = async () => {
  const run = await finishedRun()
  await setupPanel().getByLabel('Training run overlay').selectOption(run.run_id)
  await settle(5000)
  await page.getByRole('button', { name: 'Choose visible fields' }).click()
  await page.getByRole('button', { name: 'High-loss samples' }).click()
  await settle(1500)
  await page.getByRole('button', { name: 'Close field selection' }).click()
  await settle(1500)
}

const closeRunDetail = async () => {
  const close = page.locator('[aria-label="Close"], [aria-label="Close run"]').last()
  if (await close.isVisible().catch(() => false)) await close.click()
  await settle(800)
}

const filtered = async () => {
  await page.getByRole('button', { name: 'View experiments' }).click()
  await page.locator('tbody tr').first().waitFor()
  await settle(2500)
  await closeRunDetail()
  await shot('experiments-filtered')
}

const tab = name => page.locator('.modal-box').getByRole('button', { name, exact: true })

/** The newest finished run trained on `source`, waiting for it to finish if needed. */
const finishedRun = async (source = PART_1_SOURCE, timeoutMs = 25 * 60 * 1000) => {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const { runs } = await (await page.request.get(`${URL}/api/runs?max_results=20`)).json()
    const run = runs.find(candidate => candidate.magnidata['magnidata.dataset_source'] === source
      && ['FINISHED', 'FAILED', 'KILLED'].includes(candidate.status))
    if (run) {
      if (run.status !== 'FINISHED') throw new Error(`run ended ${run.status}`)
      return run
    }
    await settle(5000)
  }
  throw new Error('training did not finish in time')
}

const stages = {
  async source() {
    await home(SOURCE_URL)
    await shot('workspace-source')
  },

  async byod() {
    await page.goto(EMPTY_URL)
    await page.evaluate(() => localStorage.clear())
    await page.goto(EMPTY_URL)
    await page.getByRole('button', { name: 'Add dataset' }).first().click()
    await page.getByText('Load Demo').first().click()
    await page.getByRole('dialog').getByText('AgriStress-500').first().waitFor()
    await settle(500)
    await shot('byod-demo')
  },

  async workspace() {
    await home()
    await shot('workspace-lineage')
    await node(PART_1).click()
    await page.locator('.dataset-panel').waitFor()
    await settle(1200)
    await shot('panel-statistics')
    await page.getByRole('tab', { name: 'Images' }).click()
    await page.waitForFunction(() => [...document.querySelectorAll('.dataset-panel__images img')].every(img => img.complete && img.naturalWidth > 0))
    await shot('panel-images')
    await page.getByRole('tab', { name: 'Annotations' }).click()
    await settle(500)
    await shot('panel-annotations')
    await node(PART_2).click({ modifiers: ['Shift'] })
    await page.locator('#merge-name').waitFor()
    await page.locator('#merge-name').fill('AgriStress-500 all tiles')
    await settle(900)
    await shot('merge')
  },

  async explorer() {
    await openExplorer()
    await shot('explorer')
    await openPreview()
    await imagesLoaded()
    await shot('preview-image', page.locator('.modal-box'))
    await tab('overlay').click()
    await imagesLoaded()
    await settle(600)
    await shot('preview-overlay', page.locator('.modal-box'))
  },

  async train() {
    await openExplorer()
    await page.getByRole('button', { name: 'Train model' }).click()
    const dialog = page.getByRole('dialog', { name: 'Train a model' })
    await dialog.waitFor()
    const field = label => dialog.locator('label', { hasText: new RegExp(`^${label}`) }).locator('input').first()
    await field('Run name').fill('tiny-unet-baseline')
    await field('Epochs').fill('5')
    await settle(600)
    await shot('train-dialog', dialog)
    await dialog.getByRole('button', { name: 'Start training' }).click()
    await dialog.waitFor({ state: 'hidden' })
    await settle(25000)
    await shot('run-live', page.locator('header, .app-stage > div').first())
    console.log('  waiting for the run to finish…')
    await finishedRun()
  },

  async runs() {
    await home()
    await page.getByRole('button', { name: 'Experiments' }).first().click()
    await page.locator('tbody tr').first().waitFor()
    await page.setViewportSize({ width: 1920, height: 900 })
    await settle(1500)
    await page.screenshot({ path: path.join(OUT, 'experiments.png'), clip: { x: 0, y: 0, width: 1920, height: 330 } })
    console.log('  captured experiments')
    await page.setViewportSize({ width: 1440, height: 900 })
    await page.locator('tbody tr').first().click()
    await settle(2500)
    await shot('run-detail')
  },

  async selection() {
    await openExplorer()
    await brushHardTiles()
    await setupPanel().getByPlaceholder('Name for the new dataset').fill('Hard tiles')
    await settle(500)
    await shot('save-selection')
  },

  async loop() {
    await openExplorer()
    await brushHardTiles()
    await setupPanel().getByPlaceholder('Name for the new dataset').fill('Hard tiles')
    await setupPanel().getByRole('button', { name: /Save .* samples as dataset/ }).click()
    await page.getByRole('button', { name: 'View in workspace' }).click()
    await page.locator('.react-flow__node').filter({ hasText: 'Hard tiles' }).waitFor()
    await settle(1200)
    // Train on the new dataset and show the live badge while it runs.
    await node('Hard tiles').dblclick()
    await page.getByRole('heading', { name: 'Visual Explorer' }).waitFor()
    await page.locator('tbody tr').first().waitFor({ timeout: 60000 })
    await page.getByRole('button', { name: 'Train model' }).click()
    const dialog = page.getByRole('dialog', { name: 'Train a model' })
    await dialog.locator('label', { hasText: /^Run name/ }).locator('input').first().fill('tiny-unet-hard-tiles')
    await dialog.locator('label', { hasText: /^Epochs/ }).locator('input').first().fill('8')
    await dialog.getByRole('button', { name: 'Start training' }).click()
    await dialog.waitFor({ state: 'hidden' })
    await page.getByRole('button', { name: 'Workspace' }).first().click()
    await page.locator('.react-flow__node').first().waitFor()
    if (!(await page.locator('.dataset-panel').count())) await node('Hard tiles').click()
    await settle(9000)
    await shot('workspace-badges')
    console.log('  waiting for the second run…')
    for (let attempt = 0; attempt < 120; attempt += 1) {
      const { runs } = await (await page.request.get(`${URL}/api/runs?max_results=20`)).json()
      if (runs.length >= 2 && runs.every(candidate => candidate.status === 'FINISHED')) break
      await settle(5000)
    }
    await filtered()
  },

  async filtered() {
    await home()
    await node('Hard tiles').click()
    await filtered()
  },

  async own() {
    const openDialog = async () => {
      await home()
      await page.getByRole('button', { name: 'Add dataset' }).click()
      await page.getByRole('dialog').waitFor()
      await settle(500)
    }
    await openDialog()
    await shot('byod-choose')
    await page.getByRole('dialog').getByText('Prepare From Server Folder').click()
    await page.getByRole('dialog').getByText('my-field-survey').waitFor()
    await settle(600)
    await shot('byod-folder')
    await openDialog()
    await page.getByRole('dialog').getByText('Create New Dataset').click()
    await page.getByPlaceholder('My Field Survey').fill('My Field Survey')
    await settle(500)
    await shot('byod-upload')
    await openDialog()
    await page.getByRole('dialog').getByText('Prepare From Server Folder').click()
    const row = page.getByRole('dialog').locator('.byod-demo-row').filter({ hasText: 'my-field-survey' })
    await row.getByRole('button', { name: 'Prepare' }).click()
    await page.locator('.react-flow__node').filter({ hasText: 'my-field-survey' }).waitFor({ timeout: 300000 })
    await settle(1500)
    await page.locator('.react-flow__node').filter({ hasText: 'my-field-survey' }).click()
    await settle(1500)
    await shot('workspace-own')
  },

  async embeddings() {
    await home()
    await node('my-field-survey').dblclick()
    await page.getByRole('heading', { name: 'Visual Explorer' }).waitFor()
    await page.locator('tbody tr').first().waitFor({ timeout: 60000 })
    await settle(2500)
    await page.getByRole('button', { name: 'Embeddings', exact: true }).click()
    await settle(6000)
    await shot('embeddings-view')
  },

  /**
   * Needs a trainer built from the guide's worked example (a `tiny_unet_wide` preset, a `jaccard` loss term and a
   * `val_metric/pixel_accuracy` metric), one finished run of it named `extend-demo` on part 1, and an external run
   * named `segformer-b0-v1` logged with plain MLflow (the guide's custom-trainer snippet).
   */
  async extend() {
    await openExplorer()
    await page.getByRole('button', { name: 'Train model' }).click()
    const dialog = page.getByRole('dialog', { name: 'Train a model' })
    await dialog.waitFor()
    await dialog.locator('select').first().selectOption('tiny_unet_wide')
    await dialog.locator('label', { hasText: 'jaccard' }).locator('input[type=checkbox]').check()
    await settle(600)
    await shot('extend-train-dialog', dialog)
    await dialog.getByRole('button', { name: 'Cancel' }).click()

    await page.getByRole('button', { name: 'Experiments' }).first().click()
    await page.locator('tbody tr').filter({ hasText: 'extend-demo' }).first().waitFor()
    await page.locator('tbody tr').filter({ hasText: 'extend-demo' }).first().click()
    await settle(2500)
    await page.getByText('pixel_accuracy').first().scrollIntoViewIfNeeded()
    await settle(1500)
    await shot('extend-run-metrics')
    await page.getByRole('button', { name: 'Columns' }).first().click()
    await settle(1000)
    await shot('extend-columns')

    await openExplorer()
    const { runs } = await (await page.request.get(`${URL}/api/runs?max_results=50`)).json()
    const demo = runs.find(candidate => candidate.name === 'extend-demo')
    await setupPanel().getByLabel('Training run overlay').selectOption(demo.run_id)
    await settle(5000)
    await page.getByRole('button', { name: 'Choose visible fields' }).click()
    await page.getByPlaceholder(/Search fields/).fill('jaccard')
    await settle(1500)
    await shot('extend-axes')

    // A run logged by a trainer of your own (see the guide), with metrics the portal has never heard of.
    await page.getByRole('button', { name: 'Experiments' }).first().click()
    await page.locator('tbody tr').filter({ hasText: 'segformer-b0-v1' }).first().click()
    await settle(2500)
    await page.getByText('seconds_per_image').first().scrollIntoViewIfNeeded()
    await settle(1200)
    await shot('extend-external')
  },

  /**
   * Needs a stack built with the opt-in AgriTune override (docker-compose.agritune.yml), the fixture dataset registered
   * as "Field survey fixture", and one finished run named `agritune-smoke` of the `agritune_mlp_probe` preset on it.
   */
  async agritune() {
    await home(AGRITUNE_URL)
    await node(AGRITUNE_TITLE).dblclick()
    await page.getByRole('heading', { name: 'Visual Explorer' }).waitFor()
    await page.locator('tbody tr').first().waitFor({ timeout: 60000 })
    await settle(2500)
    await page.getByRole('button', { name: 'Train model' }).click()
    const dialog = page.getByRole('dialog', { name: 'Train a model' })
    await dialog.waitFor()
    await dialog.locator('select').first().selectOption('agritune_mlp_probe')
    await settle(800)
    await shot('agritune-train-dialog', dialog)
    await page.setViewportSize({ width: 390, height: 844 })
    await settle(800)
    await shot('agritune-train-dialog-phone')
    await page.setViewportSize({ width: 1440, height: 900 })
    await dialog.getByRole('button', { name: 'Cancel' }).click()

    const { runs } = await (await page.request.get(`${AGRITUNE_URL}/api/runs?max_results=50`)).json()
    const smoke = runs.find(candidate => candidate.name === 'agritune-smoke')
    await setupPanel().getByLabel('Training run overlay').selectOption(smoke.run_id)
    await settle(5000)
    await page.getByRole('button', { name: 'Choose visible fields' }).click()
    await page.getByPlaceholder(/Search fields/).fill('dice')
    await settle(1500)
    await shot('agritune-axes')
    await page.getByRole('button', { name: 'Close field selection' }).click()
    await settle(800)
    await openPreview(AGRITUNE_TILE, AGRITUNE_SOURCE)
    await imagesLoaded()
    await tab('prediction').click()
    await imagesLoaded()
    await settle(800)
    await shot('agritune-prediction', page.locator('.modal-box'))
    await page.keyboard.press('Escape')
  },

  async overlay() {
    await openExplorer()
    const run = await finishedRun()
    await page.getByRole('complementary', { name: 'Visual Explorer configuration' }).getByLabel('Training run overlay').selectOption(run.run_id)
    await settle(6000)
    await shot('run-overlay')
    await openPreview()
    await imagesLoaded()
    await tab('prediction').click()
    const infer = page.getByRole('button', { name: 'Run inference' })
    if (await infer.isVisible().catch(() => false)) {
      await infer.click()
      await page.locator('.modal-box').getByText('Model prediction').waitFor({ timeout: 180000 })
    }
    await imagesLoaded()
    await settle(800)
    await shot('preview-prediction', page.locator('.modal-box'))
  },
}

for (const name of wanted.length ? wanted : Object.keys(stages)) {
  if (!stages[name]) throw new Error(`unknown stage ${name}; stages: ${Object.keys(stages).join(', ')}`)
  console.log(`stage ${name}`)
  await stages[name]()
}
await browser.close()
