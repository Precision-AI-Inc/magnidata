import raw from './descriptions.json'

export interface ColDesc {
  group: string
  plain_meaning: string
  scale: string
}

// Extend the JSON with derived columns not in the original file. Keep this in sync with
// the MagniData feature extractor's output schema so the Schema tab documents every column
// that can appear as a parallel-coordinates axis.
const EXTRA: Record<string, ColDesc> = {
  cluster: {
    group: 'Final scores',
    plain_meaning:
      'A cluster/partition label carried through from the input dataset (e.g. a prior grouping of similar images). When present, the 3D view auto-colors and can partition points by this column.',
    scale: 'Text or number (label)',
  },
  cluster_l2: {
    group: 'Final scores',
    plain_meaning:
      'A second-level cluster/partition label carried through from the input dataset, alongside cluster.',
    scale: 'Text or number (label)',
  },

  // ── Learned image-quality scores (pyiqa) ──
  nima_ava: {
    group: 'Photo quality',
    plain_meaning:
      'Overall aesthetic quality from the NIMA model (Inception backbone, trained on the AVA photo-rating set). Higher means the image looks more like a well-composed, pleasing photo.',
    scale: '~1–10 (higher = better)',
  },
  nima_vgg16_ava: {
    group: 'Photo quality',
    plain_meaning:
      'Aesthetic quality from the NIMA model with a VGG16 backbone — a second opinion on overall photo appeal. Compare with nima_ava.',
    scale: '~1–10 (higher = better)',
  },
  niqe: {
    group: 'Photo quality',
    plain_meaning:
      'Blind technical-quality score (NIQE). Measures how far the image strays from the statistics of natural, undistorted photos — no training labels used. The least domain-biased quality score.',
    scale: 'Lower = better (typically 2–10)',
  },
  brisque: {
    group: 'Photo quality',
    plain_meaning:
      'No-reference technical-quality score (BRISQUE). Detects distortions such as blur, noise and compression artifacts. Lower means fewer artifacts.',
    scale: 'Lower = better (0–100)',
  },

  // ── Deterministic image-quality panel ──
  noise_sigma: {
    group: 'Photo quality',
    plain_meaning:
      'Estimated image noise/grain level (no-reference). Higher means a noisier capture — often from high ISO or low light. Tends to track gain/ISO.',
    scale: 'Higher = noisier (intensity units)',
  },
  tenengrad: {
    group: 'Photo quality',
    plain_meaning:
      'Sharpness based on edge-gradient energy. Higher means crisper focus; very low values indicate blur.',
    scale: 'Higher = sharper',
  },
  rms_contrast: {
    group: 'Photo quality',
    plain_meaning:
      'Overall contrast — the spread of brightness across the image. Low looks flat or hazy; high looks punchy.',
    scale: '0–1 (higher = more contrast)',
  },
  dynamic_range: {
    group: 'Photo quality',
    plain_meaning:
      'Spread between the darkest and brightest tones (1st–99th percentile). Low values indicate clipping or washed-out / low-light captures.',
    scale: '0–1 (higher = wider range)',
  },
  luma_entropy: {
    group: 'Photo quality',
    plain_meaning:
      'Information content of the brightness histogram. Higher means richer tonal variation; very low means a flat, featureless image.',
    scale: 'Bits, 0–8 (higher = richer)',
  },
  colorfulness: {
    group: 'Object colors',
    plain_meaning:
      'How vivid and varied the colors are (Hasler–Süsstrunk). Higher means more saturated, colorful scenes; low means muted or near-grayscale.',
    scale: 'Higher = more colorful',
  },

  // ── Coverage / camera metadata ──
  // `camera` is not an extractor output column — useCSV.ts derives it at load time by
  // parsing the image filename, so it is documented here rather than in schema.py.
  camera: {
    group: 'Camera & sensor info',
    plain_meaning:
      'The camera model extracted from the image filename. Identifies which sensor captured the photo — useful for spotting quality or style differences between camera types.',
    scale: 'Text (label)',
  },
  bg_coverage: {
    group: 'Background',
    plain_meaning:
      'Fraction of the image labeled as background (non-object) in the segmentation.',
    scale: '0–1 (share of pixels)',
  },
  gsd: {
    group: 'Camera & sensor info',
    plain_meaning:
      'Ground sample distance — the real-world size each pixel covers. Smaller means finer spatial detail.',
    scale: 'Meters per pixel (smaller = finer)',
  },
  camera_angle: {
    group: 'Camera & sensor info',
    plain_meaning:
      'Camera viewing geometry: nadir (straight down), oriented (off-nadir / oblique), or missing if unknown.',
    scale: 'Category: nadir / oriented / missing',
  },

  // ── Optional domain-specific passthrough metadata ──
  domain_metric: {
    group: 'Camera & sensor info',
    plain_meaning:
      "An optional dataset-specific metric carried through from the source dataset's own metadata, when present (its meaning depends on the dataset — e.g. a density or condition figure from the data collection site).",
    scale: 'Text / number (label)',
  },
}

// ── Training columns (merged in from an MLflow run, not part of the feature extractor) ──
const TRAINING = 'Training'
const TERM_NAMES: Record<string, string> = {
  total: 'the weighted total loss', dice: 'the Dice loss', cce: 'the cross-entropy loss', bce: 'the foreground/background BCE loss',
}
const SUFFIXES: { suffix: string; text: (term: string) => string; scale: string }[] = [
  { suffix: '', text: t => `Per-image ${t} at the selected training epoch, from an eval-mode pass without augmentation.`, scale: '≥ 0 (higher = harder)' },
  { suffix: '_mean', text: t => `Average of ${t} over every scored epoch up to the selected one: how hard the image is across training, not just now.`, scale: '≥ 0 (higher = persistently harder)' },
  { suffix: '_delta', text: t => `First scored value of ${t} minus the current one. Positive means the model improved on this image.`, scale: 'positive = improved' },
  { suffix: '_slope', text: t => `Trend of ${t} over the last 5 scored epochs (least squares). Negative means the image is getting easier.`, scale: 'negative = getting easier' },
  { suffix: '_rank', text: t => `Percentile rank of ${t} among images of the same split at the selected epoch. 1 is the hardest image.`, scale: '0–1 (1 = hardest in split)' },
]

function trainingDescs(): Record<string, ColDesc> {
  const out: Record<string, ColDesc> = {
    hardness: {
      group: TRAINING,
      plain_meaning: 'Share of scored epochs in which the image was among the hardest 20% of its split by total loss. High values mean the image stays hard however long the model trains: a candidate for relabelling or removal.',
      scale: '0–1 (higher = persistently hard)',
    },
    iou: { group: TRAINING, plain_meaning: 'Mean intersection-over-union of the predicted and labelled mask for this image at the selected epoch.', scale: '0–1 (higher = better)' },
    confidence: { group: TRAINING, plain_meaning: 'Average of the model\'s highest class probability over the image\'s pixels at the selected epoch. Low means the model is unsure.', scale: '0–1 (higher = more confident)' },
    split: { group: TRAINING, plain_meaning: 'Whether the image was in the train, validation or test split of the training run. Loss ranks are computed within each split.', scale: 'train / val / test' },
  }
  for (const [term, name] of Object.entries(TERM_NAMES)) {
    for (const { suffix, text, scale } of SUFFIXES) out[`loss_${term}${suffix}`] = { group: TRAINING, plain_meaning: text(name), scale }
  }
  return out
}

/** Description of a column, including training-run columns of loss terms not listed above (e.g. a custom `loss_focal_rank`). */
export function colDesc(col: string): ColDesc | undefined {
  if (COL_DESCS[col]) return COL_DESCS[col]
  const m = /^loss_([a-z][a-z0-9_]*?)(_mean|_delta|_slope|_rank)?$/.exec(col)
  if (!m) return undefined
  const suffix = SUFFIXES.find(x => x.suffix === (m[2] ?? ''))!
  return { group: TRAINING, plain_meaning: suffix.text(`the ${m[1]} loss`), scale: suffix.scale }
}

export const COL_DESCS: Record<string, ColDesc> = {
  ...(raw.columns as Record<string, ColDesc>),
  ...EXTRA,
  ...trainingDescs(),
}

export const GROUP_DESCS: Record<string, string> = raw.groups as Record<string, string>

export const GROUPS_ORDER = [
  'Basic info',
  'How hard to outline',
  'How crowded',
  'Object types',
  'Photo quality',
  'Object colors',
  'Background',
  'Object shapes',
  'Annotation coverage',
  'Camera & sensor info',
  'Final scores',
  'Training',
]

// Palette for group badges — a coordinated, muted family anchored by the brand
// teal/emerald/navy, with desaturated warm accents kept distinguishable across
// the 11 groups (still legible as dark text on a light tint).
export const GROUP_COLORS: Record<string, string> = {
  'Basic info':            '#5b7a8c',   // slate-teal
  'How hard to outline':   '#c2603f',   // muted terracotta
  'How crowded':           '#cf9445',   // muted ochre
  'Object types':          '#5a9e3f',   // leaf green
  'Photo quality':         '#0e8fa6',   // bright teal
  'Object colors':         '#7d6bb0',   // muted violet
  'Background':            '#8a6d3b',   // earth brown
  'Object shapes':         '#b85c8a',   // muted rose
  'Annotation coverage':   '#1aa179',   // emerald
  'Camera & sensor info':  '#006A7C',   // brand teal
  'Final scores':          '#013755',   // brand navy
  'Training':              '#b0530f',   // burnt orange: set apart from the dataset feature groups
}
