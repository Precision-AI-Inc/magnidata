import { api } from '../api'

export type PreviewTab = 'image' | 'mask' | 'overlay' | 'prediction'

export interface PreviewSourceInput {
  tab: PreviewTab
  imgPath: string
  thumbPath?: string
  baseName: string
  alpha: number
  /** The overlaid training run, if any. */
  runId?: string
  /** Epoch of the saved prediction to show; null when the run has none for this image. */
  predictionEpoch: number | null
}

/**
 * URL for the preview tab, or '' when that tab has nothing to show.
 *
 * Every tab maps to its own source and none falls back to another: the Mask and Overlay tabs show the
 * dataset's ground-truth annotation, the Prediction tab shows only a model prediction saved by a run. A
 * missing prediction must never be replaced by the ground truth, or the two would be indistinguishable.
 */
export function previewSource(input: PreviewSourceInput): string {
  const { tab, imgPath, thumbPath, baseName, alpha, runId, predictionEpoch } = input
  if (!imgPath) return ''
  switch (tab) {
    case 'prediction':
      return runId && predictionEpoch != null ? api.runPredictionUrl(runId, baseName, predictionEpoch, alpha) : ''
    case 'image':
      return api.imageUrl(thumbPath || imgPath, 1280)
    case 'mask':
      return api.annotationMaskUrl(imgPath, 1280)
    case 'overlay':
      return api.annotationOverlayUrl(imgPath, alpha, 1280)
  }
}

/** Short caption naming where the displayed mask comes from. */
export function previewCaption(tab: PreviewTab, predictionEpoch: number | null): string | null {
  if (tab === 'mask' || tab === 'overlay') return 'Ground truth · dataset annotation'
  if (tab === 'prediction') return predictionEpoch != null ? `Model prediction · epoch ${predictionEpoch}` : null
  return null
}
