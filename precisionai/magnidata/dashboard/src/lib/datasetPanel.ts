import type { DatasetMeta, DatasetSummary } from '../api'

export function formatBytes(bytes: number | undefined): string {
  if (bytes == null) return '—'
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`
  if (bytes >= 1_000) return `${Math.round(bytes / 1_000)} KB`
  return `${bytes} B`
}

export interface MaskCoverage {
  /** Share of the checked sample that has a segmentation mask, 0..1 (null when nothing was checked). */
  fraction: number | null
  label: string
}

/** Mask coverage is measured on an evenly spaced sample of rows, so it is reported as an estimate. */
export function maskCoverage(summary: Pick<DatasetSummary, 'mask_coverage'>): MaskCoverage {
  const { checked, with_masks } = summary.mask_coverage
  if (!checked) return { fraction: null, label: 'No images to check' }
  const fraction = with_masks / checked
  return { fraction, label: `${Math.round(fraction * 100)}% of ${checked} sampled images have masks` }
}

export interface LineageLinks {
  parents: DatasetMeta[]
  children: DatasetMeta[]
}

export function parentSources(dataset: DatasetMeta): string[] {
  return dataset.parent_sources ?? (dataset.parent_source ? [dataset.parent_source] : [])
}

export function lineageLinks(dataset: DatasetMeta, datasets: DatasetMeta[]): LineageLinks {
  const parents = new Set(parentSources(dataset))
  return {
    parents: datasets.filter(other => parents.has(other.source)),
    children: datasets.filter(other => parentSources(other).includes(dataset.source)),
  }
}

/** Heights (0..1) for a sparkline of histogram bins. */
export function histogramHeights(bins: number[]): number[] {
  const peak = Math.max(...bins, 0)
  return peak ? bins.map(count => count / peak) : bins.map(() => 0)
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (Number.isInteger(value)) return value.toLocaleString()
  return Math.abs(value) >= 1000 ? Math.round(value).toLocaleString() : Number(value.toPrecision(3)).toString()
}
