import { useRef, useEffect, useState, useMemo } from 'react'
import { api } from '../api'
import type { EmbeddingVariant } from '../api'
import type { CSSProperties } from 'react'
import * as THREE from 'three'
import { OrbitControls }   from 'three/examples/jsm/controls/OrbitControls.js'
import { EffectComposer }  from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass }      from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass }      from 'three/examples/jsm/postprocessing/OutputPass.js'
import type { CSVData, CSVRow } from '../types'
import type { Partitions } from '../hooks/usePartitions'
import { imageNameOf } from '../hooks/usePartitions'
import { BRAND_CATEGORICAL, BRAND_SEQ_STOPS, CLUSTER_PALETTE_100 } from '../data/vizColors'

// ── Palette & colorscale ───────────────────────────────────────────────────────
const PALETTE = BRAND_CATEGORICAL

// Brand teal → emerald → mint sequential scale (shared with ParallelCoords via vizColors)
const WARM_STOPS = BRAND_SEQ_STOPS

// ── GLSL shaders ───────────────────────────────────────────────────────────────

// Data-point particles — crisp anti-aliased circle, opaque
const POINT_VERT = /* glsl */`
  attribute float aSize;
  attribute vec3  aColor;
  varying   vec3  vColor;
  void main() {
    vColor = aColor;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (300.0 / -mv.z);
    gl_Position  = projectionMatrix * mv;
  }
`
const POINT_FRAG = /* glsl */`
  varying vec3 vColor;
  void main() {
    vec2  uv = gl_PointCoord - 0.5;
    float d  = length(uv);
    if (d > 0.5) discard;
    float aa = 1.0 - smoothstep(0.40, 0.50, d);
    gl_FragColor = vec4(vColor, aa);
  }
`

// Cluster centroid marker — filled dot with a clear ring (classic target icon)
const CENTROID_VERT = /* glsl */`
  attribute vec3  aColor;
  uniform   float uTime;
  varying   vec3  vColor;
  void main() {
    vColor = aColor;
    vec4  mv    = modelViewMatrix * vec4(position, 1.0);
    float pulse = 1.0 + 0.08 * sin(uTime * 1.8);
    gl_PointSize = 0.20 * pulse * (300.0 / -mv.z);
    gl_Position  = projectionMatrix * mv;
  }
`
const CENTROID_FRAG = /* glsl */`
  varying vec3 vColor;
  void main() {
    vec2  uv  = gl_PointCoord - 0.5;
    float d   = length(uv) * 2.0;
    if (d > 1.0) discard;
    float dot  = 1.0 - smoothstep(0.00, 0.28, d);
    float ring = smoothstep(0.52, 0.60, d) * (1.0 - smoothstep(0.70, 0.78, d));
    float aa   = 1.0 - smoothstep(0.88, 1.00, d);
    float alpha = (dot + ring * 0.85) * aa;
    gl_FragColor = vec4(vColor, min(alpha, 1.0));
  }
`

// Cluster connection lines — dim at point end, brighter toward centroid
const LINE_VERT = /* glsl */`
  attribute float aT;
  attribute vec3  aColor;
  varying   float vT;
  varying   vec3  vColor;
  void main() {
    vT     = aT;
    vColor = aColor;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`
const LINE_FRAG = /* glsl */`
  uniform float uTime;
  varying float vT;
  varying vec3  vColor;
  void main() {
    float base  = mix(0.04, 0.28, vT * vT);
    float pulse = 0.04 * sin(uTime * 1.6 + vT * 8.0);
    gl_FragColor = vec4(vColor, clamp(base + pulse, 0.0, 1.0));
  }
`

// ── Color utilities ────────────────────────────────────────────────────────────
function hexToRgb01(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1,3), 16) / 255,
    parseInt(hex.slice(3,5), 16) / 255,
    parseInt(hex.slice(5,7), 16) / 255,
  ]
}

// 100 pre-shuffled distinct colours (RGB 0..1), indexed by category/cluster id.
const CLUSTER_RGB: [number, number, number][] = CLUSTER_PALETTE_100.map(hexToRgb01)

function interpolateWarm(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  for (let i = 0; i < WARM_STOPS.length - 1; i++) {
    const a = WARM_STOPS[i], b = WARM_STOPS[i + 1]
    if (t <= b.t) {
      const s = (t - a.t) / (b.t - a.t)
      return [
        (a.r + s * (b.r - a.r)) / 255,
        (a.g + s * (b.g - a.g)) / 255,
        (a.b + s * (b.b - a.b)) / 255,
      ]
    }
  }
  const l = WARM_STOPS[WARM_STOPS.length - 1]
  return [l.r / 255, l.g / 255, l.b / 255]
}

// ── Math utilities ─────────────────────────────────────────────────────────────
function standardize(matrix: number[][]): number[][] {
  const n = matrix.length, d = matrix[0]?.length ?? 0
  if (n === 0 || d === 0) return matrix
  const means = Array<number>(d).fill(0)
  const stds  = Array<number>(d).fill(1)
  for (let j = 0; j < d; j++) {
    means[j] = matrix.reduce((s, r) => s + (r[j] ?? 0), 0) / n
    const v  = matrix.reduce((s, r) => s + ((r[j] ?? 0) - means[j]) ** 2, 0) / n
    stds[j]  = Math.sqrt(v) || 1
  }
  return matrix.map(r => r.map((v, j) => ((v ?? 0) - means[j]) / stds[j]))
}

// PCA → first 3 principal components via implicit power iteration on XᵀX. The d×d covariance
// matrix is never materialized (each step is an Xv then Xᵀ(Xv) matvec), so this stays fast and
// memory-light even for high-dimensional embedding vectors (d in the hundreds/thousands).
function pca3(matrix: number[][], iters = 50): [number, number, number][] {
  const X = standardize(matrix)
  const n = X.length, d = X[0]?.length ?? 0
  if (n < 2 || d < 1) return X.map(() => [0, 0, 0])
  const comps: number[][] = []
  for (let c = 0; c < Math.min(3, d); c++) {
    let v = Array.from({ length: d }, () => Math.random() - 0.5)
    const orthogonalize = (vec: number[] | Float64Array) => {
      for (const p of comps) {
        let dot = 0; for (let j = 0; j < d; j++) dot += vec[j] * p[j]
        for (let j = 0; j < d; j++) vec[j] -= dot * p[j]
      }
    }
    orthogonalize(v)
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1
    for (let j = 0; j < d; j++) v[j] /= norm
    for (let it = 0; it < iters; it++) {
      const Xv = new Float64Array(n)
      for (let k = 0; k < n; k++) { const row = X[k]; let s = 0; for (let j = 0; j < d; j++) s += row[j] * v[j]; Xv[k] = s }
      const w = new Float64Array(d)
      for (let k = 0; k < n; k++) { const row = X[k], xv = Xv[k]; for (let j = 0; j < d; j++) w[j] += row[j] * xv }
      orthogonalize(w)                                       // deflate against earlier components
      norm = Math.sqrt(w.reduce((s, x) => s + x * x, 0)) || 1
      const nv = new Array<number>(d)
      for (let j = 0; j < d; j++) nv[j] = w[j] / norm
      v = nv
    }
    comps.push(v)
  }
  while (comps.length < 3) comps.push(new Array<number>(d).fill(0))
  return X.map(row => {
    let a = 0, b = 0, cc = 0
    for (let j = 0; j < d; j++) { a += comps[0][j] * row[j]; b += comps[1][j] * row[j]; cc += comps[2][j] * row[j] }
    return [a, b, cc] as [number, number, number]
  })
}

// Spherical k-means — clusters in the FULL feature space by cosine distance (1 − cosine similarity).
// Vectors are L2-normalized to the unit sphere, so cosine similarity == dot product and centroid
// updates are the normalized mean of their members. `standardize` z-scores each dimension first
// (used for heterogeneously-scaled multiparametric data; embeddings are clustered as-is).
function kmeansCosine(
  rawData: number[][], k: number,
  opts: { standardize?: boolean } = {}, maxIter = 60,
): number[] {
  const n = rawData.length, d = rawData[0]?.length ?? 0
  k = Math.max(1, Math.min(k, n))
  if (n === 0 || d === 0) return new Array<number>(n).fill(0)

  // Optional per-dimension standardization
  let data = rawData
  if (opts.standardize) {
    const mean = new Array<number>(d).fill(0)
    const std  = new Array<number>(d).fill(0)
    for (const r of data) for (let j = 0; j < d; j++) mean[j] += r[j] ?? 0
    for (let j = 0; j < d; j++) mean[j] /= n
    for (const r of data) for (let j = 0; j < d; j++) std[j] += ((r[j] ?? 0) - mean[j]) ** 2
    for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / n) || 1
    data = data.map(r => r.map((v, j) => ((v ?? 0) - mean[j]) / std[j]))
  }

  // L2-normalize every row → unit vectors (cosine similarity becomes a plain dot product)
  const U = data.map(r => {
    let nrm = 0; for (let j = 0; j < d; j++) nrm += r[j] * r[j]
    nrm = Math.sqrt(nrm) || 1
    const u = new Array<number>(d)
    for (let j = 0; j < d; j++) u[j] = r[j] / nrm
    return u
  })
  const dot = (a: number[], b: number[]) => { let s = 0; for (let j = 0; j < d; j++) s += a[j] * b[j]; return s }

  // k-means++ seeding using cosine distance (1 − similarity to nearest centroid)
  const centroids: number[][] = [U[Math.floor(Math.random() * n)].slice()]
  for (let c = 1; c < k; c++) {
    const dists = U.map(p => {
      let maxSim = -Infinity
      for (const cen of centroids) { const s = dot(cen, p); if (s > maxSim) maxSim = s }
      return Math.max(0, 1 - maxSim)
    })
    const sum = dists.reduce((s, x) => s + x, 0)
    let r = Math.random() * sum, idx = 0
    for (; idx < n - 1 && r > 0; idx++) r -= dists[idx]
    centroids.push(U[idx].slice())
  }

  // Lloyd iterations: assign by max cosine similarity, recompute normalized mean centroids
  let labels = new Array<number>(n).fill(0)
  for (let iter = 0; iter < maxIter; iter++) {
    const next = U.map(p => {
      let best = 0, bestSim = -Infinity
      for (let c = 0; c < centroids.length; c++) { const s = dot(centroids[c], p); if (s > bestSim) { bestSim = s; best = c } }
      return best
    })
    if (next.every((l, i) => l === labels[i])) break
    labels = next
    for (let c = 0; c < k; c++) {
      const m = new Array<number>(d).fill(0)
      let cnt = 0
      for (let i = 0; i < n; i++) { if (labels[i] !== c) continue; const p = U[i]; for (let j = 0; j < d; j++) m[j] += p[j]; cnt++ }
      if (!cnt) continue
      let nrm = 0; for (let j = 0; j < d; j++) { m[j] /= cnt; nrm += m[j] * m[j] }
      nrm = Math.sqrt(nrm) || 1
      for (let j = 0; j < d; j++) m[j] /= nrm
      centroids[c] = m
    }
  }
  return labels
}

function normalizeToBuffer(coords: [number, number, number][]): Float32Array {
  if (!coords.length) return new Float32Array(0)
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity
  for (const [x, y, z] of coords) {
    if (x < minX) minX = x; if (x > maxX) maxX = x
    if (y < minY) minY = y; if (y > maxY) maxY = y
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
  }
  const scale = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2
  const buf = new Float32Array(coords.length * 3)
  coords.forEach(([x, y, z], i) => {
    buf[i*3]   = ((x - cx) / scale) * 2
    buf[i*3+1] = ((y - cy) / scale) * 2
    buf[i*3+2] = ((z - cz) / scale) * 2
  })
  return buf
}

// ── Position data ──────────────────────────────────────────────────────────────
export type LayoutMode = 'pca' | 'cluster'

type PositionData = {
  positions:         Float32Array
  clusterLabels?:    number[]
  clusterCentroids?: Float32Array   // k × 3, same normalized space as positions
}

// Feature matrix for the 3D layout. Either the CSV numeric columns (multiparametric) or the
// per-row embedding vectors. Rows are kept in their original order so point index == row index
// (required for click / cluster selection to map back to rows).
function buildFeatureMatrix(
  rows: CSVRow[], data: CSVData, useEmbeddings: boolean,
  embeddings: Record<string, number[]> | null, embedKeyCol: string | null,
): number[][] {
  if (useEmbeddings && embeddings && embedKeyCol) {
    let dim = 0
    for (const r of rows) { const e = embeddings[String(r[embedKeyCol])]; if (e && e.length) { dim = e.length; break } }
    if (!dim) return []
    const zero = new Array<number>(dim).fill(0)
    return rows.map(r => {
      const e = embeddings[String(r[embedKeyCol])]
      return e && e.length === dim ? e : zero      // rows without an embedding sit at the origin
    })
  }
  // Exclude the cluster column itself — it's a label, not a feature, and would distort the layout.
  const cols = data.numericCols.filter(c => c !== data.clusterCol).slice(0, 30)
  if (!cols.length) return []
  return rows.map(r => cols.map(c => (r[c] as number) ?? 0))
}

// Lay out points from precomputed cluster labels: pull each toward its (3D-projected) centroid.
// Labels come from full-space cosine clustering; this only handles the visual placement.
// Cluster count is derived from the labels themselves (not a `k` arg) and invalid labels fall back to
// the point's own position — so a momentarily-stale labels array can never index out of bounds.
function clusterPositions(proj: [number, number, number][], labels: number[]): PositionData {
  const n = proj.length
  let maxLabel = 0
  for (let i = 0; i < n; i++) { const l = labels[i]; if (typeof l === 'number' && l > maxLabel) maxLabel = l }
  const kk = maxLabel + 1
  const rawCens = Array.from({ length: kk }, () => [0, 0, 0])
  const counts  = Array<number>(kk).fill(0)
  proj.forEach(([x, y, z], i) => {
    const c = labels[i]
    if (c == null || c < 0 || c >= kk) return
    rawCens[c][0] += x; rawCens[c][1] += y; rawCens[c][2] += z; counts[c]++
  })
  rawCens.forEach((cen, i) => {
    if (counts[i]) { cen[0] /= counts[i]; cen[1] /= counts[i]; cen[2] /= counts[i] }
  })
  const rawPts = proj.map(([x, y, z], i): [number, number, number] => {
    const c = labels[i]
    const [cx, cy, cz] = (c != null && c >= 0 && c < kk) ? rawCens[c] : [x, y, z]
    return [
      cx + (x - cx) * 0.22 + (Math.random() - 0.5) * 0.06,
      cy + (y - cy) * 0.22 + (Math.random() - 0.5) * 0.06,
      cz + (z - cz) * 0.22 + (Math.random() - 0.5) * 0.06,
    ]
  })
  const allCoords = [...rawPts, ...(rawCens as [number, number, number][])]
  const normAll   = normalizeToBuffer(allCoords)
  return {
    positions:        normAll.slice(0, n * 3),
    clusterLabels:    labels,
    clusterCentroids: normAll.slice(n * 3),
  }
}

// ── Color buffer ───────────────────────────────────────────────────────────────
const SIZE_NORMAL    = 0.055
const SIZE_DIM       = 0.028
const SIZE_HIGHLIGHT = 0.16    // clicked node + its paired counterpart
const COMPARE_SIZE   = 0.08
const COMPARE_HL     = 0.20

// Muted slate for unassigned points in the cluster-by-color view.
const UNCLUSTERED_RGB: [number, number, number] = [0.30, 0.32, 0.40]

// ── gist_ncar colormap (matplotlib) ──────────────────────────────────────────────
// Piecewise-linear control points (position → channel value) for the high-variety gist_ncar map.
// Used to give each cluster a distinct, vivid colour in the cluster-by-color view.
const GIST_NCAR_R: [number, number][] = [
  [0.0, 0.0], [0.3098, 0.0], [0.3725, 0.0], [0.4235, 0.0], [0.5333, 1.0],
  [0.7922, 1.0], [0.8471, 1.0], [0.898, 0.9412], [1.0, 0.9412],
]
const GIST_NCAR_G: [number, number][] = [
  [0.0, 0.0], [0.051, 0.3922], [0.1059, 1.0], [0.1569, 0.8314], [0.1647, 0.8314],
  [0.2157, 1.0], [0.2588, 1.0], [0.2706, 0.9333], [0.3176, 0.6275], [0.3686, 0.0],
  [0.4275, 0.0], [0.5216, 1.0], [0.6314, 1.0], [0.6863, 0.851], [0.7843, 0.6],
  [0.8667, 0.851], [0.949, 1.0], [1.0, 1.0],
]
const GIST_NCAR_B: [number, number][] = [
  [0.0, 0.502], [0.051, 0.0], [0.1098, 0.0], [0.2627, 0.0], [0.3216, 1.0],
  [0.4157, 1.0], [0.4745, 1.0], [0.5333, 0.0], [0.5804, 0.0], [0.6314, 0.0],
  [0.7922, 0.0], [0.8471, 0.0], [0.898, 0.5647], [1.0, 1.0],
]
function interpSeg(seg: [number, number][], t: number): number {
  if (t <= seg[0][0]) return seg[0][1]
  for (let i = 1; i < seg.length; i++) {
    if (t <= seg[i][0]) {
      const [p0, v0] = seg[i - 1], [p1, v1] = seg[i]
      return p1 > p0 ? v0 + (v1 - v0) * ((t - p0) / (p1 - p0)) : v1
    }
  }
  return seg[seg.length - 1][1]
}
function gistNcar(t: number): [number, number, number] {
  return [interpSeg(GIST_NCAR_R, t), interpSeg(GIST_NCAR_G, t), interpSeg(GIST_NCAR_B, t)]
}

// Build `k` distinct cluster colours: sample gist_ncar at evenly spaced positions (avoiding the very
// dark/near-white ends) with a random global offset, then shuffle so each cluster gets a random hue.
// Regenerated whenever the cluster count changes, so the assignment reshuffles as k grows/shrinks.
// Assign each cluster id a colour from the 100-colour palette (id mod 100) — distinct and
// stable. Beyond 100 clusters colours necessarily repeat, but that is far past usefulness.
function makeClusterColors(k: number): [number, number, number][] {
  if (k <= 0) return []
  return Array.from({ length: k }, (_, i) => CLUSTER_RGB[i % CLUSTER_RGB.length])
}

function buildColors(
  rows: CSVRow[], data: CSVData, colorBy: string, selectedIdxs: Set<number>,
  clusterColoring?: number[] | null, clusterColors?: [number, number, number][],
): Float32Array {
  const buf       = new Float32Array(rows.length * 3)
  const hasFilter = selectedIdxs.size < rows.length

  // Cluster-by-color: tint each selected point by its cluster's (randomised gist_ncar) colour.
  // Points outside the parallel-coords / 3D filter stay grey; unassigned points use a muted slate.
  if (clusterColoring && clusterColors) {
    rows.forEach((r, i) => {
      const selected = !hasFilter || selectedIdxs.has(r._idx as number)
      if (!selected) { buf[i*3] = 0.22; buf[i*3+1] = 0.24; buf[i*3+2] = 0.32; return }
      const lab = clusterColoring[i]
      const rgb = (lab >= 0 && clusterColors[lab]) ? clusterColors[lab] : UNCLUSTERED_RGB
      buf[i*3] = rgb[0]; buf[i*3+1] = rgb[1]; buf[i*3+2] = rgb[2]
    })
    return buf
  }

  const isCat     = data.categoricalCols.includes(colorBy)
  const enc       = isCat ? data.catEncodings[colorBy] : null
  // One distinct colour per category from the 100-colour palette (id mod 100).
  const catColors = (isCat && enc) ? CLUSTER_RGB : null
  let cmin = 0, cmax = 1
  if (!isCat) {
    // 2nd–98th percentile so outliers don't compress the scale — matches the PC legend exactly
    const vals: number[] = []
    for (const r of rows) {
      const v = r[colorBy] as number
      if (v != null && isFinite(v)) vals.push(v)
    }
    if (vals.length) {
      vals.sort((a, b) => a - b)
      const lo = Math.max(0, Math.round(0.02 * (vals.length - 1)))
      const hi = Math.min(vals.length - 1, Math.round(0.98 * (vals.length - 1)))
      cmin = vals[lo]; cmax = vals[hi]
      if (cmin >= cmax) { cmin = vals[0]; cmax = vals[vals.length - 1] }
    }
  }
  rows.forEach((r, i) => {
    const selected = !hasFilter || selectedIdxs.has(r._idx as number)
    if (!selected) {
      buf[i*3] = 0.22; buf[i*3+1] = 0.24; buf[i*3+2] = 0.32
      return
    }
    let rgb: [number, number, number]
    if (isCat && enc && catColors && catColors.length) {
      rgb = catColors[(enc.encode[String(r[colorBy] ?? '')] ?? 0) % catColors.length]
    } else {
      const t = cmax > cmin ? ((r[colorBy] as number ?? cmin) - cmin) / (cmax - cmin) : 0
      rgb = interpolateWarm(t)
    }
    buf[i*3] = rgb[0]; buf[i*3+1] = rgb[1]; buf[i*3+2] = rgb[2]
  })
  return buf
}

// ── Camera fit ─────────────────────────────────────────────────────────────────
function fitCamera(camera: THREE.PerspectiveCamera, controls: OrbitControls, pos: Float32Array) {
  const n = pos.length / 3
  if (!n) return
  let cx = 0, cy = 0, cz = 0
  for (let i = 0; i < n; i++) { cx += pos[i*3]; cy += pos[i*3+1]; cz += pos[i*3+2] }
  cx /= n; cy /= n; cz /= n
  let maxR = 0
  for (let i = 0; i < n; i++) {
    const dx = pos[i*3]-cx, dy = pos[i*3+1]-cy, dz = pos[i*3+2]-cz
    maxR = Math.max(maxR, Math.sqrt(dx*dx + dy*dy + dz*dz))
  }
  const fov  = camera.fov * Math.PI / 180
  const dist = (maxR / Math.sin(fov / 2)) * 0.65 + 0.8
  camera.position.set(cx + maxR * 0.25, cy + maxR * 0.15, cz + dist)
  controls.target.set(cx, cy, cz)
  controls.update()
}

// ── Cluster overlay geometry ───────────────────────────────────────────────────
function buildClusterOverlay(
  positions: Float32Array, centroids: Float32Array,
  labels: number[], colors: Float32Array,
): { lines: THREE.LineSegments, lineMat: THREE.ShaderMaterial,
    cenPoints: THREE.Points, cenMat: THREE.ShaderMaterial } {
  const n = labels.length
  const k = centroids.length / 3

  // Cluster average colors (for centroid nodes and line centroid-ends)
  const cenColors = new Float32Array(k * 3)
  const cenCounts = new Float32Array(k)
  for (let i = 0; i < n; i++) {
    const c = labels[i]
    cenColors[c*3+0] += colors[i*3+0]
    cenColors[c*3+1] += colors[i*3+1]
    cenColors[c*3+2] += colors[i*3+2]
    cenCounts[c]++
  }
  for (let c = 0; c < k; c++) {
    if (cenCounts[c] > 0) {
      cenColors[c*3+0] /= cenCounts[c]
      cenColors[c*3+1] /= cenCounts[c]
      cenColors[c*3+2] /= cenCounts[c]
    }
  }

  // Line geometry — n segments, each [point → centroid]
  const linePosArr = new Float32Array(n * 6)
  const lineColArr = new Float32Array(n * 6)
  const lineTArr   = new Float32Array(n * 2)
  for (let i = 0; i < n; i++) {
    const c = labels[i]
    // Point vertex (T=0, dim end)
    linePosArr[i*6+0] = positions[i*3+0]; linePosArr[i*6+1] = positions[i*3+1]; linePosArr[i*6+2] = positions[i*3+2]
    lineColArr[i*6+0] = colors[i*3+0];    lineColArr[i*6+1] = colors[i*3+1];    lineColArr[i*6+2] = colors[i*3+2]
    lineTArr[i*2]     = 0
    // Centroid vertex (T=1, bright end)
    linePosArr[i*6+3] = centroids[c*3+0]; linePosArr[i*6+4] = centroids[c*3+1]; linePosArr[i*6+5] = centroids[c*3+2]
    lineColArr[i*6+3] = cenColors[c*3+0]; lineColArr[i*6+4] = cenColors[c*3+1]; lineColArr[i*6+5] = cenColors[c*3+2]
    lineTArr[i*2+1]   = 1
  }
  const lineGeo = new THREE.BufferGeometry()
  lineGeo.setAttribute('position', new THREE.BufferAttribute(linePosArr, 3).setUsage(THREE.DynamicDrawUsage))
  lineGeo.setAttribute('aColor',   new THREE.BufferAttribute(lineColArr, 3).setUsage(THREE.DynamicDrawUsage))
  lineGeo.setAttribute('aT',       new THREE.BufferAttribute(lineTArr, 1))
  const lineMat = new THREE.ShaderMaterial({
    vertexShader: LINE_VERT, fragmentShader: LINE_FRAG,
    uniforms: { uTime: { value: 0 } },
    transparent: true, depthWrite: false,
  })
  const lines = new THREE.LineSegments(lineGeo, lineMat)
  // Endpoints follow the animated particles every frame (see animate()'s per-frame position
  // sync below) without ever recomputing the geometry's bounding sphere, so a culling check
  // against it would use a stale volume — see the `points` object below for the full story.
  lines.frustumCulled = false

  // Centroid points geometry
  const cenGeo = new THREE.BufferGeometry()
  cenGeo.setAttribute('position', new THREE.BufferAttribute(centroids.slice(), 3))
  cenGeo.setAttribute('aColor',   new THREE.BufferAttribute(cenColors, 3).setUsage(THREE.DynamicDrawUsage))
  const cenMat = new THREE.ShaderMaterial({
    vertexShader: CENTROID_VERT, fragmentShader: CENTROID_FRAG,
    uniforms: { uTime: { value: 0 } },
    transparent: true, depthWrite: false,
  })
  const cenPoints = new THREE.Points(cenGeo, cenMat)
  cenPoints.frustumCulled = false   // same reasoning as `lines` above

  return { lines, lineMat, cenPoints, cenMat }
}

// ── Three state ────────────────────────────────────────────────────────────────
type ThreeState = {
  renderer:         THREE.WebGLRenderer
  scene:            THREE.Scene
  camera:           THREE.PerspectiveCamera
  controls:         OrbitControls
  composer:         EffectComposer
  // GPU particle cloud
  points:           THREE.Points
  posAttr:          THREE.BufferAttribute
  colorAttr:        THREE.BufferAttribute
  sizeAttr:         THREE.BufferAttribute
  // Cluster overlay (null when not in cluster mode)
  clusterLines:     THREE.LineSegments | null
  clusterLineMat:   THREE.ShaderMaterial | null
  centroidPoints:   THREE.Points | null
  centroidMat:      THREE.ShaderMaterial | null
  // Cluster topology (needed to update line positions each frame)
  clusterLabels:    number[] | null
  clusterCentroids: Float32Array | null
  // Comparison embeddings overlay (orange cloud + per-item link lines)
  comparePoints:    THREE.Points | null
  compareMat:       THREE.ShaderMaterial | null
  compareOrigIdx:   number[] | null         // matched original row index per comparison point
  compareRowToIdx:  Map<number, number> | null   // reverse: original row → comparison point index
  linkLines:        THREE.LineSegments | null
  linkMat:          THREE.LineBasicMaterial | null
  linkOrigIdx:      number[] | null         // original row index per link segment
  // Animation buffers
  curPos:           Float32Array
  tgtPos:           Float32Array
  curSize:          Float32Array
  tgtSize:          Float32Array
  n:                number
  animId:           number
  clock:            THREE.Clock
  lastInteract:     number
  firstLoad:        boolean
}

// ── Component ──────────────────────────────────────────────────────────────────
interface Props {
  data:             CSVData
  rows:             CSVRow[]
  embeddingsAvailable?: boolean       // API has embeddings for this dataset
  datasetSource?:   string | null     // dataset CSV path — used for server-side embedding compute
  selectedRows:     CSVRow[]
  colorBy:          string
  onClusterSelect?: (rowIndices: Set<number> | null) => void
  onPointClick?:    (row: CSVRow) => void
  partitions?:      Partitions      // persist the current clustering as cluster_id annotations
  onCommitClusterColumn?: (updates: { idx: number; value: number }[]) => void  // write the CSV cluster column on commit
}

type FeatureSource    = 'multiparametric' | 'embeddings'
type ReductionMethod  = 'pca' | 'tsne' | 'lle'
const REDUCTION_LABELS: Record<ReductionMethod, string> = { pca: 'PCA', tsne: 't-SNE', lle: 'LLE' }

// How clusters are rendered: 'spread' separates them spatially; 'color' keeps the projection layout
// and tints each point by its cluster id.
type ClusterView = 'spread' | 'color'

// Segmented control to switch how clusters are rendered: spatial Spread vs. in-place Color.
function ClusterViewToggle({ value, onChange, btnS, lblS }: {
  value: ClusterView
  onChange: (v: ClusterView) => void
  btnS: (active: boolean) => CSSProperties
  lblS: CSSProperties
}) {
  const options: { id: ClusterView; label: string; title: string }[] = [
    { id: 'spread', label: 'Spread', title: 'Separate clusters spatially (pull points toward their centroids)' },
    { id: 'color',  label: 'Color',  title: 'Keep the projection layout and tint each point by its cluster id' },
  ]
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={lblS}>View</span>
      {options.map(o => (
        <button key={o.id} onClick={() => onChange(o.id)} style={btnS(value === o.id)} title={o.title}>
          {o.label}
        </button>
      ))}
    </span>
  )
}

export function Graph3D({ data, rows, embeddingsAvailable, datasetSource, selectedRows, colorBy, onClusterSelect, onPointClick, partitions, onCommitClusterColumn }: Props) {
  const mountRef = useRef<HTMLDivElement>(null)
  const threeRef = useRef<ThreeState | null>(null)

  // Keep these fresh inside the mount-effect closure (avoids stale captures)
  const onClusterSelectRef = useRef(onClusterSelect)
  const onPointClickRef = useRef(onPointClick)
  const rowsRef = useRef(rows)
  useEffect(() => { onClusterSelectRef.current = onClusterSelect }, [onClusterSelect])
  useEffect(() => { onPointClickRef.current = onPointClick }, [onPointClick])
  useEffect(() => { rowsRef.current = rows }, [rows])

  const [mode,  setMode]  = useState<LayoutMode>('pca')
  const [k,     setK]     = useState(20)
  const [featureSource, setFeatureSource] = useState<FeatureSource>('multiparametric')
  const [reduction, setReduction] = useState<ReductionMethod>('pca')

  // ── Comparison embeddings (a second model, computed server-side) ─────────────
  const [variants, setVariants]           = useState<EmbeddingVariant[]>([])
  const [compareVariant, setCompareVariant] = useState<string>('')        // '' = none
  const [linkEnabled, setLinkEnabled]     = useState(false)
  // Clicked node + its paired counterpart, both enlarged to make the connection explicit
  const [highlight, setHighlight] = useState<{ row: number; cmp: number } | null>(null)

  const embeddingsReady = !!embeddingsAvailable
  const useEmbeddings    = featureSource === 'embeddings' && embeddingsReady

  // Prefer embeddings automatically when available; fall back to multiparametric otherwise.
  useEffect(() => {
    setFeatureSource(embeddingsReady ? 'embeddings' : 'multiparametric')
  }, [embeddingsReady])

  // Discover sibling comparison-embedding files (<stem>_<model>.json) for the loaded dataset.
  useEffect(() => {
    if (!datasetSource || !embeddingsReady) { setVariants([]); return }
    let alive = true
    api.datasetEmbeddingVariants(datasetSource)
      .then(v => { if (alive) setVariants(v) })
      .catch(() => { if (alive) setVariants([]) })
    return () => { alive = false }
  }, [datasetSource, embeddingsReady])

  // Reset the comparison selection whenever the dataset changes.
  useEffect(() => { setCompareVariant(''); setLinkEnabled(false) }, [datasetSource])

  const selectedIdxs = useMemo(
    () => new Set(selectedRows.map(r => r._idx as number)),
    [selectedRows],
  )

  // Feature matrix for MULTIPARAMETRIC mode only (small CSV numerics, computed client-side).
  // Embedding-space math (projection, clustering) runs entirely on the API.
  const featureMatrix = useMemo<number[][]>(
    () => (useEmbeddings ? [] : buildFeatureMatrix(rows, data, false, null, null)),
    [rows, data, useEmbeddings],
  )

  // 3D projection (display positions only). For embeddings the API computes PCA/t-SNE/LLE and returns
  // the positions; multiparametric reduces the small matrix client-side (PCA). Off the render path
  // with a "computing" flag for the overlay.
  const [projection, setProjection] = useState<[number, number, number][]>([])
  const [projecting, setProjecting] = useState(false)
  useEffect(() => {
    let cancelled = false
    if (useEmbeddings && datasetSource) {
      setProjecting(true)
      // A transient failure here (network blip, or the server's first request after startup
      // paying a one-off cold-import cost) used to permanently strand the view with every
      // point sitting at the origin — nothing here would ever retry, so only a full page
      // reload (a fresh fetch) recovered. Retry a couple of times with backoff before giving
      // up, so a one-off hiccup self-heals instead of requiring a manual refresh.
      const load = async () => {
        const retryDelaysMs = [500, 1500]
        for (let attempt = 0; ; attempt++) {
          try {
            const r = await api.embeddingProjection(datasetSource, reduction)
            if (!cancelled) { setProjection(r.positions as [number, number, number][]); setProjecting(false) }
            return
          } catch (e) {
            if (cancelled) return
            if (attempt >= retryDelaysMs.length) { setProjection([]); setProjecting(false); throw e }
            await new Promise(res => setTimeout(res, retryDelaysMs[attempt]))
          }
        }
      }
      load().catch(() => {})
      return () => { cancelled = true }
    }
    if (!featureMatrix.length || !featureMatrix[0]?.length) {
      setProjection(rows.map(() => [0, 0, 0] as [number, number, number]))
      setProjecting(false)
      return
    }
    setProjecting(true)
    const id = setTimeout(() => {                                  // defer so the overlay can paint
      const result = pca3(featureMatrix)
      if (!cancelled) { setProjection(result); setProjecting(false) }
    }, 16)
    return () => { cancelled = true; clearTimeout(id) }
  }, [featureMatrix, reduction, useEmbeddings, datasetSource]) // eslint-disable-line

  // Cluster labels (cosine spherical k-means on the full feature space). Embeddings → API; else client.
  const [clusterLabels, setClusterLabels] = useState<number[] | null>(null)
  useEffect(() => {
    let cancelled = false
    if (mode !== 'cluster') { setClusterLabels(null); return }
    if (useEmbeddings && datasetSource) {
      api.embeddingClusters(datasetSource, k)
        .then(r => { if (!cancelled) setClusterLabels(r.labels) })
        .catch(() => { if (!cancelled) setClusterLabels(null) })
      return () => { cancelled = true }
    }
    if (!featureMatrix.length || !featureMatrix[0]?.length) { setClusterLabels(null); return }
    setClusterLabels(kmeansCosine(featureMatrix, k, { standardize: true }))
    return () => { cancelled = true }
  }, [featureMatrix, mode, k, useEmbeddings, datasetSource]) // eslint-disable-line

  // ── Commit ("make partition"): snapshot the current k-means clustering onto the filtered selection ──
  // Writes the CSV cluster column (1-based ids: "Cluster 1, 2, …") — the ONLY thing that modifies the
  // column — and also persists the partitions to SQLite. Re-running with a different K and committing
  // again overwrites the ids.
  const [committing, setCommitting] = useState(false)
  const [commitMsg,  setCommitMsg]  = useState<string | null>(null)
  const commitPartitions = async () => {
    if (!clusterLabels) return
    const updates: { idx: number; value: number }[] = []
    const entries: { image_name: string; cluster_id: number }[] = []
    for (const r of selectedRows) {
      const label = clusterLabels[r._idx as number]
      if (label == null || label < 0) continue
      updates.push({ idx: r._idx as number, value: label + 1 })
      if (data.imagePath) {
        const name = imageNameOf(r, data)
        if (name) entries.push({ image_name: name, cluster_id: label + 1 })
      }
    }
    if (!updates.length) return
    setCommitting(true); setCommitMsg(null)
    try {
      onCommitClusterColumn?.(updates)                                   // modify the CSV cluster column
      if (partitions && entries.length) await partitions.commit(entries) // persist (SQLite / export)
      setCommitMsg(`${updates.length.toLocaleString()} partitioned`)
    } catch (e) {
      setCommitMsg(`error: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setCommitting(false)
    }
  }

  // ── Cluster label sources ─────────────────────────────────────────────────
  // A "preset" clustering comes from a CSV cluster column (preferred) or, failing that, from
  // previously-committed partitions (SQLite). The user can always switch to live k-means.
  // All label arrays are 0-based and aligned to `rows` (unassigned rows → -1).
  const partMap = partitions?.map

  // Labels read from a predetermined CSV cluster column: distinct values (string names) → 0-based ids.
  // `names` keeps the original cluster names so the UI shows the clustering exactly as labelled.
  const columnLabels = useMemo<{ labels: number[]; count: number; names: string[] } | null>(() => {
    const col = data.clusterCol
    if (!col) return null
    const present = [...new Set(rows.map(r => r[col]).filter(v => v != null && v !== '').map(v => String(v)))]
      .sort((a, b) => {
        const na = Number(a), nb = Number(b)
        if (isFinite(na) && isFinite(nb)) return na - nb
        return a.localeCompare(b)
      })
    if (!present.length) return null
    const index = new Map(present.map((v, i) => [v, i]))
    const labels = rows.map(r => {
      const v = r[col]
      return (v == null || v === '') ? -1 : (index.get(String(v)) ?? -1)
    })
    return { labels, count: present.length, names: present }
  }, [data.clusterCol, rows])

  // Labels from committed partitions (SQLite), keyed by image filename.
  const savedLabels = useMemo<number[] | null>(() => {
    if (!partMap || !data.imagePath || partMap.size === 0) return null
    let any = false
    const labels = rows.map(r => {
      const name = imageNameOf(r, data)
      const e = name ? partMap.get(name) : undefined
      if (e && e.cluster_id != null) { any = true; return e.cluster_id - 1 }
      return -1
    })
    return any ? labels : null
  }, [partMap, rows, data])

  // The predetermined clustering shown when source = 'preset' (CSV column wins over committed).
  const preset = useMemo<{ labels: number[]; count: number; names: string[]; fromColumn: boolean } | null>(() => {
    if (columnLabels) return { ...columnLabels, fromColumn: true }
    if (savedLabels) {
      let mx = -1
      for (const l of savedLabels) if (l > mx) mx = l
      const count = mx + 1
      return { labels: savedLabels, count, names: Array.from({ length: count }, (_, i) => `Cluster ${i + 1}`), fromColumn: false }
    }
    return null
  }, [columnLabels, savedLabels])
  const hasPreset = !!preset

  // Default to the preset clustering when a dataset has one; changing K flips to k-means.
  const [clusterSource, setClusterSource] = useState<'preset' | 'computed'>('computed')
  useEffect(() => { setClusterSource(hasPreset ? 'preset' : 'computed') }, [hasPreset, datasetSource])

  // Load K from the preset's cluster count (once per dataset; the user can still change it & re-run).
  // When the preset comes from a CSV cluster column, also switch to the Clusters layout to show it.
  const kInitFor = useRef<string | null>(null)
  useEffect(() => {
    if (preset && preset.count >= 2 && kInitFor.current !== data.fileName) {
      kInitFor.current = data.fileName
      setK(Math.max(2, Math.min(50, preset.count)))
      if (preset.fromColumn) setMode('cluster')
    }
  }, [preset, data.fileName])

  const [clusterView, setClusterView] = useState<ClusterView>('spread')

  const effectiveLabels = useMemo(
    () => (clusterSource === 'preset' && preset) ? preset.labels : clusterLabels,
    [clusterSource, preset, clusterLabels],
  )

  // Number of distinct clusters currently shown (drives the colour palette in the color view).
  const clusterCount = useMemo(() => {
    if (mode !== 'cluster' || !effectiveLabels) return 0
    let mx = -1
    for (const l of effectiveLabels) if (l > mx) mx = l
    return mx + 1
  }, [mode, effectiveLabels])

  // A random, distinct gist_ncar colour per cluster — regenerated whenever the cluster count changes
  // (or the color view is (re)entered) so the palette reshuffles as k grows/shrinks.
  const clusterColors = useMemo<[number, number, number][]>(
    () => (clusterView === 'color' ? makeClusterColors(clusterCount) : []),
    [clusterView, clusterCount],
  )

  // Names for the legend / cluster picker: the dataset's own cluster names when showing the preset.
  const clusterNames = useMemo<string[]>(
    () => (clusterSource === 'preset' && preset)
      ? preset.names
      : Array.from({ length: clusterCount }, (_, i) => `Cluster ${i + 1}`),
    [clusterSource, preset, clusterCount],
  )

  // ── Cluster picker (dropdown) — selects an entire cluster, same effect as Ctrl+clicking it ──
  const [pickedCluster, setPickedCluster] = useState('')
  const selectCluster = (val: string) => {
    setPickedCluster(val)
    if (val === '' || !effectiveLabels) { onClusterSelect?.(null); return }
    const k = +val
    const ids = new Set<number>()
    rows.forEach((r, i) => { if (effectiveLabels[i] === k) ids.add(r._idx as number) })
    onClusterSelect?.(ids.size ? ids : null)
  }
  // Reset the picker whenever the selection is cleared elsewhere (e.g. the "clear 3D selection" badge).
  useEffect(() => { if (selectedRows.length >= rows.length) setPickedCluster('') }, [selectedRows.length, rows.length])

  const posData = useMemo<PositionData>(() => {
    const n = rows.length
    if (n === 0) return { positions: new Float32Array(0) }
    // Projection/labels are async — until they match the CURRENT row count they're stale (a dataset
    // switch or K change still in flight). Emit n×3 zeros so positions always match the GPU buffers.
    if (projection.length !== n) return { positions: new Float32Array(n * 3) }
    // 'spread' view pulls points toward their cluster centroids; 'color' view keeps the projection
    // layout (cluster identity is conveyed by point colour instead — see `colors`).
    if (mode === 'cluster' && clusterView === 'spread' && effectiveLabels && effectiveLabels.length === n) {
      return clusterPositions(projection, effectiveLabels)
    }
    return { positions: normalizeToBuffer(projection) }   // pca, cluster-by-color
  }, [rows, mode, clusterView, useEmbeddings, projection, effectiveLabels]) // eslint-disable-line

  const colors = useMemo(() => {
    // Cluster-by-color view tints each point by its cluster id; every other view colours by `colorBy`.
    const clusterColoring = (mode === 'cluster' && clusterView === 'color'
      && effectiveLabels && effectiveLabels.length === rows.length) ? effectiveLabels : null
    return buildColors(rows, data, colorBy, selectedIdxs, clusterColoring, clusterColors)
  }, [rows, data, colorBy, selectedIdxs, mode, clusterView, effectiveLabels, clusterColors]) // eslint-disable-line

  // ── Comparison: the API returns name-matched neighbours mapped to original rows ──────────────
  const [compareCalc, setCompareCalc] = useState<{ nbr: number[][]; w: number[][]; origIdx: number[] } | null>(null)
  const [comparing, setComparing] = useState(false)
  useEffect(() => {
    if (!useEmbeddings || !datasetSource || !compareVariant) { setCompareCalc(null); setComparing(false); return }
    let cancelled = false
    setComparing(true)
    api.embeddingCompare(datasetSource, compareVariant)
      .then(r => { if (!cancelled) { setCompareCalc(r.nbr.length ? r : null); setComparing(false) } })
      .catch(() => { if (!cancelled) { setCompareCalc(null); setComparing(false) } })
    return () => { cancelled = true }
  }, [useEmbeddings, datasetSource, compareVariant])

  // Interpolate each comparison item's neighbour positions into the current layout (cheap).
  const compareData = useMemo<{ positions: Float32Array; origIdx: number[] } | null>(() => {
    if (!compareCalc || !posData.positions.length) return null
    const { nbr, w, origIdx } = compareCalc
    const m = nbr.length
    const P = posData.positions
    const np = P.length / 3
    const pos = new Float32Array(m * 3)
    for (let i = 0; i < m; i++) {
      let x = 0, y = 0, z = 0
      const ns = nbr[i], ws = w[i]
      for (let t = 0; t < ns.length; t++) {
        const o = ns[t]
        if (o < 0 || o >= np) continue
        const wt = ws[t]; x += P[o*3]*wt; y += P[o*3+1]*wt; z += P[o*3+2]*wt
      }
      pos[i*3] = x + (Math.random() - 0.5) * 0.02
      pos[i*3+1] = y + (Math.random() - 0.5) * 0.02
      pos[i*3+2] = z + (Math.random() - 0.5) * 0.02
    }
    return { positions: pos, origIdx }
  }, [compareCalc, posData])

  // ── Mount — rebuild when dataset size changes ─────────────────────────────
  useEffect(() => {
    const el = mountRef.current
    if (!el) return
    const n = rows.length

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x000000)
    renderer.toneMapping = THREE.ReinhardToneMapping
    renderer.toneMappingExposure = 1.0
    el.appendChild(renderer.domElement)

    const scene  = new THREE.Scene()
    scene.fog    = new THREE.FogExp2(0x000000, 0.055)

    const camera = new THREE.PerspectiveCamera(55, el.clientWidth / (el.clientHeight || 1), 0.001, 100)
    camera.position.set(0, 0, 4)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.minDistance   = 0.3
    controls.maxDistance   = 20
    controls.addEventListener('start', () => {
      if (threeRef.current) threeRef.current.lastInteract = Date.now()
    })

    // ── Click-to-select cluster / point ───────────────────────────────────
    let downX = 0, downY = 0
    const onMouseDown = (e: MouseEvent) => { downX = e.clientX; downY = e.clientY }
    const onCanvasClick = (e: MouseEvent) => {
      if ((e.clientX - downX) ** 2 + (e.clientY - downY) ** 2 > 25) return  // drag, not click
      const s = threeRef.current
      if (!s) return
      s.lastInteract = Date.now()
      const ctrl = e.ctrlKey || e.metaKey   // Ctrl (or ⌘) → select/illuminate the whole cluster

      const rect = renderer.domElement.getBoundingClientRect()
      const ndc  = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width)  *  2 - 1,
        -((e.clientY - rect.top) / rect.height) *  2 + 1,
      )
      const raycaster = new THREE.Raycaster()
      raycaster.params.Points = { threshold: 0.07 }
      s.scene.updateMatrixWorld()
      // Positions animate in-place after mount, but the geometry's bounding sphere was cached from
      // the initial origin positions. Refresh it so the ray/sphere early-out doesn't drop every hit.
      s.points.geometry.computeBoundingSphere()
      const targets: THREE.Object3D[] = [s.points]
      if (s.comparePoints) { s.comparePoints.geometry.computeBoundingSphere(); targets.push(s.comparePoints) }
      raycaster.setFromCamera(ndc, s.camera)
      const hits = raycaster.intersectObjects(targets, false)

      if (!hits.length) {
        if (ctrl) onClusterSelectRef.current?.(null)   // Ctrl+click on empty space clears the 3D selection
        else setHighlight(null)                         // plain click on empty space clears the highlight
        return
      }

      const currentRows = rowsRef.current
      const hit = hits[0]
      const isCompare = !!s.comparePoints && hit.object === s.comparePoints

      // ── Ctrl+click → cluster selection ONLY (no preview, no pair highlight) ──
      if (ctrl) {
        const rowIdx = isCompare ? (s.compareOrigIdx ? s.compareOrigIdx[hit.index!] : -1) : hit.index!
        if (rowIdx < 0) return
        if (s.clusterLabels && s.clusterLabels[rowIdx] >= 0) {
          const cluster = s.clusterLabels[rowIdx]
          const ids = new Set<number>()
          currentRows.forEach((r, i) => { if (s.clusterLabels![i] === cluster) ids.add(r._idx as number) })
          onClusterSelectRef.current?.(ids)
        } else {
          const rid = currentRows[rowIdx]?._idx
          if (rid != null) onClusterSelectRef.current?.(new Set([rid as number]))
        }
        return
      }

      // ── Plain click → preview the image + enlarge the clicked node and its paired counterpart ──
      if (isCompare) {
        const ci = hit.index!
        const oi = s.compareOrigIdx ? s.compareOrigIdx[ci] : -1
        setHighlight({ row: oi, cmp: ci })
        const row = oi >= 0 ? currentRows[oi] : null
        if (row) onPointClickRef.current?.(row)
        return
      }
      const hitIdx = hit.index!
      const hitRow = currentRows[hitIdx]
      setHighlight({ row: hitIdx, cmp: s.compareRowToIdx?.get(hitIdx) ?? -1 })
      if (hitRow) onPointClickRef.current?.(hitRow)
    }
    renderer.domElement.addEventListener('mousedown', onMouseDown)
    renderer.domElement.addEventListener('click',     onCanvasClick)

    // ── Subtle starfield (depth reference only) ────────────────────────────
    const starPos = new Float32Array(1200 * 3)
    for (let i = 0; i < 1200 * 3; i += 3) {
      const theta = Math.random() * Math.PI * 2
      const phi   = Math.acos(2 * Math.random() - 1)
      const r     = 9 + Math.random() * 8
      starPos[i]   = r * Math.sin(phi) * Math.cos(theta)
      starPos[i+1] = r * Math.sin(phi) * Math.sin(theta)
      starPos[i+2] = r * Math.cos(phi)
    }
    const starsGeo = new THREE.BufferGeometry()
    starsGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3))
    const starsMat = new THREE.PointsMaterial({
      size: 0.025, color: 0x334466, sizeAttenuation: true, transparent: true, opacity: 0.3,
    })
    scene.add(new THREE.Points(starsGeo, starsMat))

    // ── GPU particle cloud ─────────────────────────────────────────────────
    const initPos   = new Float32Array(n * 3).fill(0)   // start at center
    const initColor = new Float32Array(n * 3)
    const initSize  = new Float32Array(n).fill(0)        // invisible at first
    for (let i = 0; i < n * 3; i += 3) { initColor[i] = 0.1; initColor[i+1] = 0.12; initColor[i+2] = 0.18 }

    const pointGeo = new THREE.BufferGeometry()
    const posAttr   = new THREE.BufferAttribute(initPos.slice(),   3).setUsage(THREE.DynamicDrawUsage)
    const colorAttr = new THREE.BufferAttribute(initColor.slice(), 3).setUsage(THREE.DynamicDrawUsage)
    const sizeAttr  = new THREE.BufferAttribute(initSize.slice(),  1).setUsage(THREE.DynamicDrawUsage)
    pointGeo.setAttribute('position', posAttr)
    pointGeo.setAttribute('aColor',   colorAttr)
    pointGeo.setAttribute('aSize',    sizeAttr)

    const pointMat = new THREE.ShaderMaterial({
      vertexShader:   POINT_VERT,
      fragmentShader: POINT_FRAG,
      transparent:    true,
      depthWrite:     false,
      depthTest:      true,
    })
    const points = new THREE.Points(pointGeo, pointMat)
    // Points start at the origin and fly out to their real (async) positions once the
    // projection/PCA result arrives (see the "Sync positions" effect below) — but Three.js
    // computes and caches a geometry's boundingSphere lazily on first use and never
    // invalidates it as the position buffer is mutated afterward. If that first render
    // happens before real positions arrive (slow projection fetch, cold server), the cached
    // sphere is a zero-radius point at the origin; once fitCamera moves the camera to frame
    // the real (now spread-out) cloud, frustum culling against that stale sphere can drop
    // the whole cloud or leave it looking collapsed to a single point — exactly the bug this
    // component's click handler already works around for raycasting (see onCanvasClick's own
    // computeBoundingSphere() call) by recomputing on demand. Disabling culling here sidesteps
    // the staleness entirely — cheap for a point count this size, and correct regardless of
    // how the async load races against the first frame.
    points.frustumCulled = false
    scene.add(points)

    // ── Bloom post-processing ──────────────────────────────────────────────
    // Subtle glow only: a low strength + high luminance threshold so dense clusters keep their
    // colour instead of blooming into a white glare. (strength, radius, threshold)
    const w0 = el.clientWidth, h0 = el.clientHeight || 1
    const composer  = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(w0, h0), 0.18, 0.3, 0.85)
    composer.addPass(bloomPass)
    composer.addPass(new OutputPass())

    // ── State ──────────────────────────────────────────────────────────────
    const state: ThreeState = {
      renderer, scene, camera, controls, composer,
      points, posAttr, colorAttr, sizeAttr,
      clusterLines: null, clusterLineMat: null,
      centroidPoints: null, centroidMat: null,
      clusterLabels: null, clusterCentroids: null,
      comparePoints: null, compareMat: null, compareOrigIdx: null, compareRowToIdx: null,
      linkLines: null, linkMat: null, linkOrigIdx: null,
      curPos:   new Float32Array(n * 3).fill(0),
      tgtPos:   new Float32Array(n * 3).fill(0),
      curSize:  new Float32Array(n).fill(0),
      tgtSize:  new Float32Array(n).fill(SIZE_NORMAL),
      n, animId: 0,
      clock: new THREE.Clock(),
      lastInteract: 0, firstLoad: true,
    }
    threeRef.current = state

    // ── Render loop ────────────────────────────────────────────────────────
    const animate = () => {
      state.animId = requestAnimationFrame(animate)
      controls.update()

      const t = state.clock.getElapsedTime()
      if (state.centroidMat) state.centroidMat.uniforms.uTime.value = t
      if (state.clusterLineMat) state.clusterLineMat.uniforms.uTime.value = t

      // ── Position lerp ──────────────────────────────────────────────────
      let posChanged = false
      for (let i = 0; i < state.n * 3; i++) {
        const diff = state.tgtPos[i] - state.curPos[i]
        if (Math.abs(diff) > 0.0003) { state.curPos[i] += diff * 0.055; posChanged = true }
        else if (state.curPos[i] !== state.tgtPos[i]) { state.curPos[i] = state.tgtPos[i] }
      }
      if (posChanged) {
        state.posAttr.array.set(state.curPos)
        state.posAttr.needsUpdate = true

        // Update cluster line point-end positions to follow moving particles
        if (state.clusterLines && state.clusterLabels) {
          const lp = state.clusterLines.geometry.attributes.position.array as Float32Array
          for (let i = 0; i < state.n; i++) {
            lp[i*6]   = state.curPos[i*3]
            lp[i*6+1] = state.curPos[i*3+1]
            lp[i*6+2] = state.curPos[i*3+2]
          }
          state.clusterLines.geometry.attributes.position.needsUpdate = true
        }

        // Keep link-line "from" ends attached to the (moving) original points
        if (state.linkLines && state.linkOrigIdx) {
          const lp = state.linkLines.geometry.attributes.position.array as Float32Array
          for (let li = 0; li < state.linkOrigIdx.length; li++) {
            const oi = state.linkOrigIdx[li]
            lp[li*6]   = state.curPos[oi*3]
            lp[li*6+1] = state.curPos[oi*3+1]
            lp[li*6+2] = state.curPos[oi*3+2]
          }
          state.linkLines.geometry.attributes.position.needsUpdate = true
        }
      }

      // ── Size lerp (selection highlight) ───────────────────────────────
      let sizeChanged = false
      for (let i = 0; i < state.n; i++) {
        const diff = state.tgtSize[i] - state.curSize[i]
        if (Math.abs(diff) > 0.0005) { state.curSize[i] += diff * 0.10; sizeChanged = true }
        else if (state.curSize[i] !== state.tgtSize[i]) { state.curSize[i] = state.tgtSize[i] }
      }
      if (sizeChanged) {
        state.sizeAttr.array.set(state.curSize)
        state.sizeAttr.needsUpdate = true
      }

      composer.render()
    }
    animate()

    // ── Resize ─────────────────────────────────────────────────────────────
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth, h = el.clientHeight || 1
      renderer.setSize(w, h); composer.setSize(w, h)
      bloomPass.setSize(w, h)
      camera.aspect = w / h; camera.updateProjectionMatrix()
    })
    ro.observe(el)
    renderer.setSize(w0, h0); composer.setSize(w0, h0)

    return () => {
      cancelAnimationFrame(state.animId)
      ro.disconnect()
      controls.dispose()
      renderer.domElement.removeEventListener('mousedown', onMouseDown)
      renderer.domElement.removeEventListener('click',     onCanvasClick)
      pointGeo.dispose(); pointMat.dispose()
      starsGeo.dispose(); starsMat.dispose()
      state.clusterLines?.geometry.dispose(); state.clusterLineMat?.dispose()
      state.centroidPoints?.geometry.dispose(); state.centroidMat?.dispose()
      state.comparePoints?.geometry.dispose(); state.compareMat?.dispose()
      state.linkLines?.geometry.dispose(); state.linkMat?.dispose()
      renderer.dispose()
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement)
      threeRef.current = null
    }
  }, [rows.length]) // eslint-disable-line

  // ── Sync positions ─────────────────────────────────────────────────────────
  useEffect(() => {
    const s = threeRef.current
    if (!s || !posData.positions.length) return

    // Update target positions
    s.tgtPos.set(posData.positions)

    if (s.firstLoad) {
      // Projection is async, so the first posData may be all-zeros. Only fit the camera once we have
      // real (non-degenerate) positions, then fly in from center.
      let maxAbs = 0
      for (let i = 0; i < posData.positions.length; i++) { const v = Math.abs(posData.positions[i]); if (v > maxAbs) maxAbs = v }
      if (maxAbs > 1e-6) {
        s.firstLoad = false
        fitCamera(s.camera, s.controls, posData.positions)
      }
    }

    // Remove old cluster overlay
    if (s.clusterLines)  { s.scene.remove(s.clusterLines);  s.clusterLines.geometry.dispose();  s.clusterLineMat!.dispose() }
    if (s.centroidPoints){ s.scene.remove(s.centroidPoints); s.centroidPoints.geometry.dispose(); s.centroidMat!.dispose() }
    s.clusterLines = null; s.clusterLineMat = null
    s.centroidPoints = null; s.centroidMat = null
    s.clusterLabels = null; s.clusterCentroids = null

    // Build new cluster overlay if in cluster mode
    if (posData.clusterLabels && posData.clusterCentroids) {
      const currentColors = s.colorAttr.array as Float32Array
      const { lines, lineMat, cenPoints, cenMat } = buildClusterOverlay(
        posData.positions, posData.clusterCentroids,
        posData.clusterLabels, currentColors,
      )
      s.clusterLines    = lines;     s.clusterLineMat   = lineMat
      s.centroidPoints  = cenPoints; s.centroidMat      = cenMat
      s.clusterLabels   = posData.clusterLabels
      s.clusterCentroids = posData.clusterCentroids
      s.scene.add(lines)
      s.scene.add(cenPoints)
    }
  }, [posData])

  // Keep the labels used for Ctrl+click cluster selection in sync for BOTH views. Runs after the
  // posData effect (which only sets them for the spread overlay) so the color view stays selectable.
  useEffect(() => {
    const s = threeRef.current
    if (!s) return
    s.clusterLabels = (mode === 'cluster' && effectiveLabels && effectiveLabels.length === rows.length)
      ? effectiveLabels : (posData.clusterLabels ?? null)
  }, [posData, effectiveLabels, mode, rows.length])

  // ── Sync colors & selection sizes ──────────────────────────────────────────
  useEffect(() => {
    const s = threeRef.current
    if (!s || !colors.length) return

    s.colorAttr.array.set(colors)
    s.colorAttr.needsUpdate = true

    const hasFilter = selectedRows.length < rows.length
    for (let i = 0; i < rows.length; i++) {
      s.tgtSize[i] = (!hasFilter || selectedIdxs.has(rows[i]._idx as number))
        ? SIZE_NORMAL : SIZE_DIM
    }
    // Enlarge the clicked original point so its paired comparison node is easy to relate
    if (highlight && highlight.row >= 0 && highlight.row < rows.length) {
      s.tgtSize[highlight.row] = SIZE_HIGHLIGHT
    }

    // Update cluster overlay colors
    if (s.clusterLines && s.clusterLabels && s.clusterCentroids) {
      const n = s.n, k = s.clusterCentroids.length / 3
      const cenColors = new Float32Array(k * 3)
      const cenCounts = new Float32Array(k)
      for (let i = 0; i < n; i++) {
        const c = s.clusterLabels[i]
        cenColors[c*3]   += colors[i*3];   cenCounts[c]++
        cenColors[c*3+1] += colors[i*3+1]
        cenColors[c*3+2] += colors[i*3+2]
      }
      for (let c = 0; c < k; c++) {
        if (cenCounts[c] > 0) {
          cenColors[c*3]   /= cenCounts[c]
          cenColors[c*3+1] /= cenCounts[c]
          cenColors[c*3+2] /= cenCounts[c]
        }
      }
      const lc = s.clusterLines.geometry.attributes.aColor.array as Float32Array
      for (let i = 0; i < n; i++) {
        const c = s.clusterLabels[i]
        lc[i*6]   = colors[i*3];   lc[i*6+1] = colors[i*3+1]; lc[i*6+2] = colors[i*3+2]
        lc[i*6+3] = cenColors[c*3]; lc[i*6+4] = cenColors[c*3+1]; lc[i*6+5] = cenColors[c*3+2]
      }
      s.clusterLines.geometry.attributes.aColor.needsUpdate = true
      if (s.centroidPoints) {
        const cc = s.centroidPoints.geometry.attributes.aColor.array as Float32Array
        cc.set(cenColors)
        s.centroidPoints.geometry.attributes.aColor.needsUpdate = true
      }
    }
  }, [colors, selectedIdxs, rows, selectedRows.length, highlight]) // eslint-disable-line

  // ── Sync comparison cloud + link lines ──────────────────────────────────────
  useEffect(() => {
    const s = threeRef.current
    if (!s) return
    if (s.comparePoints) { s.scene.remove(s.comparePoints); s.comparePoints.geometry.dispose(); s.compareMat?.dispose() }
    if (s.linkLines)     { s.scene.remove(s.linkLines);     s.linkLines.geometry.dispose();     s.linkMat?.dispose() }
    s.comparePoints = null; s.compareMat = null; s.compareOrigIdx = null; s.compareRowToIdx = null
    s.linkLines = null; s.linkMat = null; s.linkOrigIdx = null
    setHighlight(null)
    if (!compareData) return

    const m = compareData.positions.length / 3
    // Comparison points — a distinct ORANGE cloud overlaid on the originals
    const cg = new THREE.BufferGeometry()
    cg.setAttribute('position', new THREE.BufferAttribute(compareData.positions.slice(), 3))
    const cCol = new Float32Array(m * 3)
    for (let i = 0; i < m; i++) { cCol[i*3] = 1.0; cCol[i*3+1] = 0.52; cCol[i*3+2] = 0.06 }
    cg.setAttribute('aColor', new THREE.BufferAttribute(cCol, 3))
    cg.setAttribute('aSize',  new THREE.BufferAttribute(new Float32Array(m).fill(COMPARE_SIZE), 1).setUsage(THREE.DynamicDrawUsage))
    const cMat = new THREE.ShaderMaterial({ vertexShader: POINT_VERT, fragmentShader: POINT_FRAG, transparent: true, depthWrite: false })
    const cPts = new THREE.Points(cg, cMat)
    cPts.frustumCulled = false   // same staleness concern as the main `points` object
    s.comparePoints = cPts; s.compareMat = cMat; s.compareOrigIdx = compareData.origIdx
    const r2i = new Map<number, number>()
    compareData.origIdx.forEach((row, ci) => { if (row >= 0 && !r2i.has(row)) r2i.set(row, ci) })
    s.compareRowToIdx = r2i
    s.scene.add(cPts)

    // Link lines — one segment per matched item: original position → comparison position
    const linkRows: number[] = []
    for (let i = 0; i < m; i++) if (compareData.origIdx[i] >= 0) linkRows.push(i)
    if (linkRows.length) {
      const lp = new Float32Array(linkRows.length * 6)
      for (let li = 0; li < linkRows.length; li++) {
        const i = linkRows[li], oi = compareData.origIdx[i]
        lp[li*6]   = s.curPos[oi*3];   lp[li*6+1] = s.curPos[oi*3+1]; lp[li*6+2] = s.curPos[oi*3+2]
        lp[li*6+3] = compareData.positions[i*3]; lp[li*6+4] = compareData.positions[i*3+1]; lp[li*6+5] = compareData.positions[i*3+2]
      }
      const lg = new THREE.BufferGeometry()
      lg.setAttribute('position', new THREE.BufferAttribute(lp, 3).setUsage(THREE.DynamicDrawUsage))
      const lMat = new THREE.LineBasicMaterial({ color: 0xff8c1e, transparent: true, opacity: 0.3 })
      const ll = new THREE.LineSegments(lg, lMat)
      ll.frustumCulled = false   // endpoints follow curPos every frame — see `lines` above
      ll.visible = linkEnabled
      s.linkLines = ll; s.linkMat = lMat
      s.linkOrigIdx = linkRows.map(i => compareData.origIdx[i])
      s.scene.add(ll)
    }
  }, [compareData]) // eslint-disable-line

  // Toggle link-line visibility without rebuilding
  useEffect(() => {
    const s = threeRef.current
    if (s?.linkLines) s.linkLines.visible = linkEnabled
  }, [linkEnabled, compareData])

  // Enlarge the highlighted comparison point (paired counterpart of the clicked node)
  useEffect(() => {
    const s = threeRef.current
    const sizeAttr = s?.comparePoints?.geometry.attributes.aSize as THREE.BufferAttribute | undefined
    if (!sizeAttr) return
    const arr = sizeAttr.array as Float32Array
    arr.fill(COMPARE_SIZE)
    if (highlight && highlight.cmp >= 0 && highlight.cmp < arr.length) arr[highlight.cmp] = COMPARE_HL
    sizeAttr.needsUpdate = true
  }, [highlight, compareData])

  // ── UI ────────────────────────────────────────────────────────────────────────
  const hasFilter = selectedRows.length < rows.length

  const btnS = (active: boolean): CSSProperties => ({
    padding: '4px 11px', borderRadius: 5, fontSize: 11, cursor: 'pointer',
    fontWeight: 600, lineHeight: 1,
    background: active ? 'rgba(0,106,124,0.85)' : 'rgba(255,255,255,0.05)',
    color:      active ? '#fff'                  : 'rgba(150,210,205,0.65)',
    border: `1px solid ${active ? 'rgba(109,242,163,0.50)' : 'rgba(120,200,190,0.16)'}`,
    transition: 'all .15s',
  })

  const selS: CSSProperties = {
    fontSize: 10, padding: '3px 5px', borderRadius: 5, outline: 'none',
    background: '#0c2433',                 // solid dark so the option popup isn't white
    border: '1px solid rgba(120,200,190,0.30)',
    color: 'rgba(220,243,234,0.95)',
    colorScheme: 'dark',                   // render the native dropdown popup in dark mode
  }
  // Explicit option colours — belt-and-suspenders for browsers that ignore color-scheme.
  const optS: CSSProperties = { background: '#0c2433', color: '#dff3ec' }

  const lblS: CSSProperties = {
    fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
    letterSpacing: '.6px', color: 'rgba(120,205,185,0.50)',
  }

  return (
    <div style={{ position: 'relative', flex: 1, minHeight: 0, overflow: 'hidden' }}>
      <div ref={mountRef} style={{ position: 'absolute', inset: 0 }} />

      {/* Floating controls */}
      <div style={{
        position: 'absolute', top: 10, left: 10, zIndex: 10,
        background: 'rgba(1,8,22,0.84)', backdropFilter: 'blur(14px)',
        WebkitBackdropFilter: 'blur(14px)',
        border: '1px solid rgba(90,200,185,0.14)',
        borderRadius: 10, padding: '7px 10px',
        display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 7,
      }}>
        <span style={lblS}>Source</span>
        {(['multiparametric', 'embeddings'] as FeatureSource[]).map(s => {
          const disabled = s === 'embeddings' && !embeddingsReady
          return (
            <button
              key={s}
              disabled={disabled}
              onClick={() => setFeatureSource(s)}
              title={disabled ? 'No embeddings file (.json) for this dataset' : ''}
              style={{ ...btnS(featureSource === s), opacity: disabled ? 0.4 : 1,
                       cursor: disabled ? 'not-allowed' : 'pointer' }}
            >
              {s === 'multiparametric' ? 'Multiparametric' : 'Embeddings'}
            </button>
          )
        })}

        <span style={{ ...lblS, marginLeft: 4 }}>Layout</span>
        {(['pca', 'cluster'] as LayoutMode[]).map(m => (
          <button key={m} onClick={() => setMode(m)} style={btnS(mode === m)}>
            {m === 'pca' ? 'Scatter' : 'Clusters'}
          </button>
        ))}

        {/* Dimensionality-reduction method for embeddings (computed server-side) */}
        {useEmbeddings && (<>
          <span style={{ ...lblS, marginLeft: 4 }}>Projection</span>
          {(['pca', 'tsne', 'lle'] as ReductionMethod[]).map(r => (
            <button key={r} onClick={() => setReduction(r)} style={btnS(reduction === r)}
                    title={r === 'pca' ? 'Linear PCA (fast)'
                         : r === 'tsne' ? 't-SNE — non-linear, preserves local neighbourhoods'
                         : 'Locally Linear Embedding — non-linear manifold'}>
              {REDUCTION_LABELS[r]}
            </button>
          ))}
        </>)}

        {/* Cluster source — offered when the dataset has a predetermined clustering (CSV column or
            committed partitions). The button reflects where the preset comes from. */}
        {mode === 'cluster' && hasPreset && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={lblS}>Clusters</span>
            <button onClick={() => setClusterSource('preset')} style={btnS(clusterSource === 'preset')}
                    title={preset?.fromColumn
                      ? `Visualize the ground-truth clusters provided in the file's "${data.clusterCol}" column (${preset.count} clusters)`
                      : `Visualize the saved (committed) partitions (${preset?.count} clusters)`}>
              {preset?.fromColumn ? 'Ground Truth' : 'Saved'}
            </button>
            <button onClick={() => setClusterSource('computed')} style={btnS(clusterSource === 'computed')}
                    title="Re-cluster with k-means (does not modify the cluster column until you commit)">
              K-means
            </button>
          </span>
        )}

        {mode === 'cluster' && (!hasPreset || clusterSource === 'computed') && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={lblS}>K</span>
            <input type="number" min={2} max={50} value={k}
              onChange={e => { setK(Math.max(2, Math.min(50, +e.target.value))); setClusterSource('computed') }}
              style={{ ...selS, width: 44, textAlign: 'center' }} />
          </span>
        )}

        {/* Render clusters by spatial spread (current) or by colour in the projection layout */}
        {mode === 'cluster' && (
          <ClusterViewToggle value={clusterView} onChange={setClusterView} btnS={btnS} lblS={lblS} />
        )}

        {/* Pick a whole cluster from a dropdown (same effect as Ctrl+clicking it) */}
        {mode === 'cluster' && effectiveLabels && clusterNames.length > 0 && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={lblS}>Select</span>
            <select value={pickedCluster} onChange={e => selectCluster(e.target.value)} style={{ ...selS, maxWidth: 130 }}>
              <option value="" style={optS}>All clusters</option>
              {clusterNames.map((name, i) => <option key={i} value={i} style={optS}>{name}</option>)}
            </select>
          </span>
        )}

        {/* Commit ("make partition") — writes the k-means result into the CSV cluster column. */}
        {mode === 'cluster' && clusterSource === 'computed' && (onCommitClusterColumn || (partitions && data.imagePath)) && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <button
              onClick={commitPartitions}
              disabled={!clusterLabels || committing || !selectedRows.length}
              title={clusterLabels
                ? `Write the cluster id of the ${selectedRows.length.toLocaleString()} selected rows into the cluster column`
                : 'Clustering not ready'}
              style={{ ...btnS(false),
                       background: 'rgba(41,204,165,0.18)', color: 'rgba(141,231,184,0.95)',
                       border: '1px solid rgba(109,242,163,0.45)',
                       cursor: (!clusterLabels || committing || !selectedRows.length) ? 'not-allowed' : 'pointer',
                       opacity: (!clusterLabels || !selectedRows.length) ? 0.45 : 1 }}>
              {committing ? 'Partitioning…' : 'Create partitions'}
            </button>
            {commitMsg && (
              <span style={{ fontSize: 10, fontWeight: 600,
                             color: commitMsg.startsWith('error') ? '#f87171' : 'rgba(141,231,184,0.85)' }}>
                {commitMsg}
              </span>
            )}
          </span>
        )}

        {/* Comparison embeddings — overlay another model's vectors on the same items */}
        {useEmbeddings && variants.length > 0 && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 4 }}>
            <span style={lblS}>Compare</span>
            <select value={compareVariant} onChange={e => setCompareVariant(e.target.value)} style={selS}>
              <option value="">none</option>
              {variants.map(v => <option key={v.variant} value={v.variant}>{v.name}</option>)}
            </select>
            {compareCalc && (
              <button onClick={() => setLinkEnabled(v => !v)}
                      title="Draw a line from each item's original point to its comparison point"
                      style={btnS(linkEnabled)}>
                Link
              </button>
            )}
          </span>
        )}
      </div>

      {/* Cluster legend — names ↔ colours, shown in the cluster colour view */}
      {mode === 'cluster' && clusterView === 'color' && clusterColors.length > 0 && (
        <div style={{
          position: 'absolute', top: 10, right: 10, zIndex: 10, maxHeight: '60%', maxWidth: 200,
          overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 3,
          background: 'rgba(1,8,22,0.84)', backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
          border: '1px solid rgba(90,200,185,0.14)', borderRadius: 10, padding: '8px 10px',
        }}>
          <span style={{ ...lblS, marginBottom: 2 }}>
            {clusterSource === 'preset'
              ? (preset?.fromColumn ? 'Ground Truth' : 'Saved')
              : 'Clusters'} · {clusterColors.length}
          </span>
          {clusterColors.map((rgb, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 9, height: 9, borderRadius: 2, flexShrink: 0,
                             background: `rgb(${Math.round(rgb[0]*255)},${Math.round(rgb[1]*255)},${Math.round(rgb[2]*255)})` }} />
              <span style={{ fontSize: 10.5, color: 'rgba(200,232,224,0.9)', overflow: 'hidden',
                             textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={clusterNames[i] ?? `Cluster ${i + 1}`}>
                {clusterNames[i] ?? `Cluster ${i + 1}`}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Computing overlay — while the server-side projection/clustering/comparison is in flight */}
      {(projecting || comparing) && (
        <div style={{
          position: 'absolute', inset: 0, zIndex: 20, display: 'flex',
          alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 10,
          background: 'rgba(1,8,22,0.55)', backdropFilter: 'blur(2px)', pointerEvents: 'none',
        }}>
          <div style={{
            width: 26, height: 26, borderRadius: '50%',
            border: '2.5px solid rgba(109,242,163,0.25)', borderTopColor: 'rgba(109,242,163,0.95)',
            animation: 'spin 0.8s linear infinite',
          }} />
          <span style={{ fontSize: 11, fontWeight: 600, color: 'rgba(141,231,184,0.95)', letterSpacing: '.3px' }}>
            {projecting ? `Computing ${useEmbeddings ? REDUCTION_LABELS[reduction] : 'PCA'} projection…` : 'Placing comparison embeddings…'}
          </span>
        </div>
      )}

      {/* Comparison legend */}
      {compareCalc && compareVariant && (
        <div style={{
          position: 'absolute', bottom: 12, left: '50%', transform: 'translateX(-50%)', zIndex: 10,
          display: 'flex', alignItems: 'center', gap: 6,
          fontSize: 10, fontWeight: 600, padding: '3px 9px', borderRadius: 20,
          background: 'rgba(255,140,30,0.14)', border: '1px solid rgba(255,140,30,0.35)',
          color: 'rgba(255,176,92,0.95)', backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)',
        }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'rgb(255,133,15)', display: 'inline-block' }} />
          compare: {compareVariant}
          <span style={{ opacity: 0.6 }}>· click a point to verify</span>
        </div>
      )}

      {/* Selection badge */}
      {hasFilter && (
        <div style={{
          position: 'absolute', bottom: 12, left: 12, zIndex: 10,
          fontSize: 10, fontWeight: 600, padding: '3px 10px', borderRadius: 20,
          background: 'rgba(41,204,165,0.16)', border: '1px solid rgba(109,242,163,0.30)',
          color: 'rgba(141,231,184,0.95)', backdropFilter: 'blur(6px)',
          WebkitBackdropFilter: 'blur(6px)',
        }}>
          {selectedRows.length.toLocaleString()}
          <span style={{ opacity: 0.45 }}> / {rows.length.toLocaleString()}</span>
          <span style={{ marginLeft: 5, opacity: 0.65 }}>lit</span>
        </div>
      )}

      <div style={{
        position: 'absolute', bottom: 12, right: 12, zIndex: 10,
        fontSize: 9, color: 'rgba(90,195,180,0.30)',
        letterSpacing: '.3px', userSelect: 'none', pointerEvents: 'none',
      }}>
        drag · scroll · click to preview · ctrl+click cluster
      </div>
    </div>
  )
}
