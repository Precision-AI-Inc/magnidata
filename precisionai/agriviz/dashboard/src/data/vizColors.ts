/* ═══════════════════════════════════════════════════════════════════════
   Precision AI — Visualization color system
   Single source of truth shared by the 3D graph, the parallel-coordinates
   plot, and the data tables, so every data surface speaks the same brand
   color language (navy → teal → emerald → mint).
═══════════════════════════════════════════════════════════════════════ */

export interface RGBStop { t: number; r: number; g: number; b: number }

// ── Sequential scale (low → high) ──────────────────────────────────────
// Deep teal → brand teal → emerald → fresh mint → pale mint.
// Tuned to stay legible on the dark #0a1422 viz canvas, with a bright high
// end so dense overlapping lines/points still read clearly.
export const BRAND_SEQ_STOPS: RGBStop[] = [
  { t: 0,    r: 26,  g: 74,  b: 92  },   // deep teal
  { t: 0.22, r: 0,   g: 106, b: 124 },   // brand teal  (--pai-teal  #006A7C)
  { t: 0.45, r: 0,   g: 150, b: 142 },   // teal-emerald
  { t: 0.68, r: 41,  g: 204, b: 165 },   // brand green (--pai-green #29CCA5)
  { t: 0.85, r: 109, g: 242, b: 163 },   // brand fresh (--pai-fresh #6DF2A3)
  { t: 1,    r: 198, g: 255, b: 214 },   // pale mint
]

// Plotly-shaped colorscale derived from the same stops (kept in sync by code,
// never by hand) so the 2D and 3D views are guaranteed to match.
export const BRAND_SEQUENTIAL: [number, string][] =
  BRAND_SEQ_STOPS.map(s => [s.t, `rgb(${s.r},${s.g},${s.b})`])

// ── Qualitative palette (categorical encodings on dark panels) ─────────
// Brand-cohesive teal/green/mint family, with enough hue + lightness spread
// to keep categories distinguishable against the dark canvas.
export const BRAND_CATEGORICAL = [
  '#29CCA5', // emerald (brand green)
  '#6DF2A3', // fresh mint (brand)
  '#00A0B4', // bright teal
  '#9FE6C2', // sage mint
  '#0E7C90', // deep teal
  '#4FC4D6', // sky teal
  '#02AA52', // plant green (brand)
  '#B6F0CE', // pale mint
]

// ── Light-background category badges (table cells) ─────────────────────
// [background, text] pairs, brand-tinted with dark text that clears WCAG AA
// contrast against the light tint.
export const BRAND_CAT_BADGES: [string, string][] = [
  ['#E0F2F1', '#006A7C'],
  ['#D9F5E6', '#057A3C'],
  ['#DCEEF3', '#013755'],
  ['#E6F8F0', '#0E7C90'],
  ['#D4F0EC', '#00796B'],
  ['#EAF6E4', '#3F7A1E'],
  ['#E8F4F8', '#024A75'],
  ['#DFF6EA', '#02AA52'],
]

// ── 100 distinct colours for many-category encodings (clusters, color-by) ──
// Golden-angle hue rotation across 5 saturation/lightness tiers, then shuffled so
// neighbouring ids land on far-apart hues. Index by category/cluster id (mod 100).
export const CLUSTER_PALETTE_100: string[] = [
  '#B26950', '#141AC7', '#7714C7', '#4FE0AA', '#ED49F2', '#F2AD49',
  '#14C748', '#9149F2', '#8949F2', '#761694', '#94166E', '#E449F2',
  '#F25149', '#14C751', '#50B2B2', '#DCF249', '#169417', '#B2A350',
  '#C73414', '#B29F50', '#49F2C9', '#C72B14', '#91B250', '#1614C7',
  '#49B8F2', '#7A4FE0', '#321694', '#C7145E', '#8DB250', '#164594',
  '#944016', '#B2506C', '#4FC1E0', '#944616', '#8AE04F', '#C94FE0',
  '#169461', '#B250A1', '#49F2D1', '#5081B2', '#6E14C7', '#701694',
  '#E04FA9', '#80F249', '#941674', '#99C714', '#38C714', '#F2499C',
  '#5550B2', '#E04F5A', '#D04FE0', '#147BC7', '#8A50B2', '#14C7B2',
  '#C78C14', '#168994', '#948A16', '#94162A', '#50B27A', '#5950B2',
  '#E0E04F', '#F249A4', '#40C714', '#507CB2', '#57B250', '#91E04F',
  '#D9E04F', '#16945B', '#49F275', '#4964F2', '#F24949', '#78F249',
  '#8E50B2', '#168394', '#599416', '#E04F53', '#49C0F2', '#C79414',
  '#E0994F', '#50B27E', '#4FE0B1', '#E0924F', '#5CB250', '#163F94',
  '#B26E50', '#4FC8E0', '#4F72E0', '#495CF2', '#A1C714', '#4FE062',
  '#C714B6', '#941630', '#50B2AF', '#2C1694', '#C714BF', '#49F26D',
  '#E04FA2', '#948416', '#4FE05B', '#C71455',
]

// ── Table header navy (dark header strip over light table body) ────────
export const TABLE_HEADER       = '#013755'   // --pai-navy
export const TABLE_HEADER_ACTIVE = '#024a75'   // --pai-navy-mid
