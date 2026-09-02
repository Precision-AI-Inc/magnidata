import { useState, useEffect, useCallback, type CSSProperties } from 'react'
import {
  X, ChevronLeft, ChevronRight, Image as ImageIcon, Plus, Check,
  ExternalLink, AlertTriangle, RotateCw,
} from 'lucide-react'
import { api } from '../api'
import type { CSVData, CSVRow, ColMeta } from '../types'
import type { PartitionEntry } from '../hooks/usePartitions'
import { imageNameOf } from '../hooks/usePartitions'

function navBtnStyle(side: 'left' | 'right', enabled: boolean): CSSProperties {
  return {
    position: 'absolute', top: '50%', transform: 'translateY(-50%)',
    left:  side === 'left'  ? 16 : undefined,
    right: side === 'right' ? 16 : undefined,
    width: 44, height: 44, borderRadius: '50%', zIndex: 6,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'rgba(8,22,34,0.55)', color: '#fff',
    border: '1px solid rgba(255,255,255,0.18)',
    backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)',
    boxShadow: '0 8px 24px rgba(0,0,0,0.45)',
    cursor: enabled ? 'pointer' : 'default',
    opacity: enabled ? 1 : 0, pointerEvents: enabled ? 'auto' : 'none',
    transition: 'background 0.15s, opacity 0.15s',
  }
}

interface Props {
  data: CSVData
  row: CSVRow
  allRows: CSVRow[]
  onClose: () => void
  partitionMap?: Map<string, PartitionEntry>
  removed?: Set<number>                // accepted for compat; no longer used
  onToggleRemove?: (row: CSVRow) => void
  added?: Set<number>
  onToggleAdd?: (row: CSVRow) => void
}

// Columns worth surfacing as big "highlight" tiles, in priority order.
const HERO_PRIORITY = [
  'complexity_score', 'category', 'green_annotation_ratio', 'instance_count',
  'nima_ava', 'niqe', 'overlap_ratio', 'colorfulness',
]

const prettify = (s: string) => s.replace(/_/g, ' ')

type PreviewTab = 'image' | 'mask' | 'overlay'

export function ImageModal({ data, row: initialRow, allRows, onClose, partitionMap, added, onToggleAdd }: Props) {
  const [row, setRow]           = useState(initialRow)
  const [imgError, setImgError] = useState<string | null>(null)
  const [noAnnotations, setNoAnnotations] = useState(false)
  const [loading, setLoading]   = useState(true)
  const [reloadKey, setReloadKey] = useState(0)

  const [tab, setTab] = useState<PreviewTab>('image')
  // Immediate value drives the slider itself; the debounced value drives the actual
  // image request, so dragging doesn't fire a request (and a reload flicker) per pixel.
  const [sliderAlpha, setSliderAlpha]   = useState(0.5)
  const [overlayAlpha, setOverlayAlpha] = useState(0.5)
  useEffect(() => {
    const t = setTimeout(() => setOverlayAlpha(sliderAlpha), 150)
    return () => clearTimeout(t)
  }, [sliderAlpha])

  const idx     = allRows.findIndex(r => r._idx === row._idx)
  const hasPrev = idx > 0
  const hasNext = idx < allRows.length - 1

  const imgPath   = data.imagePath ? String(row[data.imagePath] ?? '') : ''
  // A dataset's own pre-generated small preview (e.g. a CDN thumbnail column) — when
  // present, the Image tab loads this instead of resizing the full-resolution original
  // (which, for a remote-hosted dataset, can be tens of MB). "Full resolution" below
  // always points at the real imgPath regardless.
  const thumbPath = data.thumbnailPath ? String(row[data.thumbnailPath] ?? '') : ''
  const stem      = data.stemCol   ? String(row[data.stemCol] ?? String(row._idx)) : String(row._idx)

  const w = Number(row['width']); const h = Number(row['height'])
  const dims = Number.isFinite(w) && Number.isFinite(h) && w && h ? `${w} × ${h}` : null

  const partImgName = imageNameOf(row, data)
  const partEntry   = partImgName ? partitionMap?.get(partImgName) : undefined

  const metaByName = new Map<string, ColMeta>(data.cols.map(c => [c.name, c]))

  // Highlight tiles: prioritized known columns, padded with leading numeric cols.
  const heroes = (() => {
    const picked = HERO_PRIORITY.filter(c => data.numericCols.includes(c))
    for (const c of data.numericCols) {
      if (picked.length >= 4) break
      if (!picked.includes(c)) picked.push(c)
    }
    return picked.slice(0, 4)
  })()
  const heroSet = new Set(heroes)
  const metricCols = data.numericCols.filter(c => !heroSet.has(c) && c !== 'width' && c !== 'height')

  useEffect(() => { setTab('image') }, [row._idx])
  useEffect(() => { setImgError(null); setNoAnnotations(false); setLoading(true) }, [row, reloadKey, tab])

  const navigate = (dir: -1 | 1) => {
    const next = allRows[idx + dir]
    if (next) setRow(next)
  }

  const isAdded = !!added?.has(row._idx as number)
  const toggleAdd = () => {
    if (!onToggleAdd) return
    const wasAdded = isAdded
    onToggleAdd(row)
    if (!wasAdded && hasNext) navigate(1)
  }

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
    if (e.key === 'ArrowLeft')  navigate(-1)
    if (e.key === 'ArrowRight') navigate(1)
    if ((e.key === 'a' || e.key === 'A') && onToggleAdd) { e.preventDefault(); toggleAdd() }
  }, [row, idx, isAdded]) // eslint-disable-line

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const imgSrc = !imgPath ? '' :
    tab === 'image'   ? api.imageUrl(thumbPath || imgPath, 1280) :
    tab === 'mask'    ? api.annotationMaskUrl(imgPath, 1280) :
                        api.annotationOverlayUrl(imgPath, overlayAlpha, 1280)
  const origUrl = imgPath ? api.imageUrl(imgPath, 4096) : ''

  return (
    <div className="modal-backdrop" onClick={e => { if (e.target === e.currentTarget) onClose() }}
         style={{ background: 'rgba(2,10,18,0.72)', backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)' }}>
      <div className="modal-box" style={{
             display: 'flex', flexDirection: 'column', maxWidth: 1180, height: '90vh', width: '95vw',
             borderRadius: 16, overflow: 'hidden', border: '1px solid var(--c-border-2)',
             boxShadow: '0 30px 80px rgba(0,0,0,0.55)',
           }}
           onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-center gap-3 px-5 shrink-0"
             style={{ height: 56, background: 'linear-gradient(90deg, var(--pai-navy), var(--pai-navy-mid))',
                      color: '#fff', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
          <span style={{ width: 30, height: 30, borderRadius: 8, display: 'flex', alignItems: 'center',
                         justifyContent: 'center', background: 'rgba(255,255,255,0.1)', flexShrink: 0 }}>
            <ImageIcon size={15} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="truncate" style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 13, fontWeight: 600 }}>{stem}</div>
            <div style={{ fontSize: 10.5, color: 'rgba(255,255,255,0.5)' }}>
              Image {idx + 1} of {allRows.length}{dims ? ` · ${dims}px` : ''}
            </div>
          </div>
          {onToggleAdd && (
            <button onClick={toggleAdd} title={isAdded ? 'Remove from basket' : 'Add to curation basket (A)'}
                    style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11.5, fontWeight: 600,
                             padding: '5px 11px', borderRadius: 8, cursor: 'pointer', whiteSpace: 'nowrap', color: '#fff',
                             background: isAdded ? 'rgba(41,204,165,0.95)' : 'rgba(255,255,255,0.1)',
                             border: `1px solid ${isAdded ? 'rgba(41,204,165,0.95)' : 'rgba(255,255,255,0.22)'}` }}>
              {isAdded ? <Check size={13} /> : <Plus size={13} />}{isAdded ? 'Added' : 'Add'}
            </button>
          )}
          <button onClick={onClose} title="Close (Esc)"
                  style={{ width: 32, height: 32, borderRadius: 8, display: 'flex', alignItems: 'center',
                           justifyContent: 'center', color: 'rgba(255,255,255,0.7)', cursor: 'pointer',
                           background: 'transparent', border: 'none', transition: 'background .15s' }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.12)' }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}>
            <X size={17} />
          </button>
        </div>

        {/* Body */}
        <div className="flex flex-1 min-h-0">

          {/* Image stage */}
          <div className="flex flex-col flex-1 min-w-0">
            {/* Image / Mask / Overlay tabs */}
            <div className="flex items-center gap-1 px-4 shrink-0"
                 style={{ height: 40, background: 'var(--pai-navy)', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              {(['image', 'mask', 'overlay'] as const).map(t => (
                <button key={t} onClick={() => setTab(t)}
                        style={{ padding: '5px 12px', borderRadius: 7, fontSize: 11.5, fontWeight: 600,
                                 cursor: 'pointer', textTransform: 'capitalize', border: 'none',
                                 background: tab === t ? 'rgba(255,255,255,0.14)' : 'transparent',
                                 color: tab === t ? '#fff' : 'rgba(255,255,255,0.55)' }}>
                  {t}
                </button>
              ))}
              {tab === 'overlay' && (
                <div className="flex items-center gap-2" style={{ marginLeft: 'auto' }}>
                  <span style={{ fontSize: 10.5, color: 'rgba(255,255,255,0.5)' }}>Opacity</span>
                  <input type="range" min={0} max={100} value={Math.round(sliderAlpha * 100)}
                         onChange={e => setSliderAlpha(Number(e.target.value) / 100)}
                         style={{ width: 110, accentColor: 'var(--pai-fresh)' }} />
                  <span style={{ fontSize: 10.5, color: 'rgba(255,255,255,0.7)', width: 32, textAlign: 'right',
                                 fontFamily: 'JetBrains Mono, monospace' }}>
                    {Math.round(sliderAlpha * 100)}%
                  </span>
                </div>
              )}
            </div>

            <div className="flex-1 flex items-center justify-center min-h-0"
                 style={{ position: 'relative', padding: 24,
                          background: 'radial-gradient(120% 100% at 50% 0%, rgba(0,136,153,0.18) 0%, transparent 60%),' +
                                      'linear-gradient(160deg, #0b1f2e 0%, #071521 100%)', backgroundColor: '#071521' }}>
              <div aria-hidden style={{ position: 'absolute', inset: 0, opacity: 0.04, pointerEvents: 'none',
                     backgroundImage: 'linear-gradient(45deg,#fff 25%,transparent 25%),linear-gradient(-45deg,#fff 25%,transparent 25%),' +
                                      'linear-gradient(45deg,transparent 75%,#fff 75%),linear-gradient(-45deg,transparent 75%,#fff 75%)',
                     backgroundSize: '22px 22px', backgroundPosition: '0 0,0 11px,11px -11px,-11px 0' }} />

              <button aria-label="Previous" disabled={!hasPrev} onClick={() => navigate(-1)} style={navBtnStyle('left', hasPrev)}
                      onMouseEnter={e => { if (hasPrev) e.currentTarget.style.background = 'rgba(0,106,124,0.85)' }}
                      onMouseLeave={e => { e.currentTarget.style.background = 'rgba(8,22,34,0.55)' }}>
                <ChevronLeft size={22} />
              </button>
              <button aria-label="Next" disabled={!hasNext} onClick={() => navigate(1)} style={navBtnStyle('right', hasNext)}
                      onMouseEnter={e => { if (hasNext) e.currentTarget.style.background = 'rgba(0,106,124,0.85)' }}
                      onMouseLeave={e => { e.currentTarget.style.background = 'rgba(8,22,34,0.55)' }}>
                <ChevronRight size={22} />
              </button>

              {imgSrc && loading && !imgError && !noAnnotations && (
                <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                              alignItems: 'center', justifyContent: 'center', gap: 12, zIndex: 4 }}>
                  <div style={{ width: 40, height: 40, borderRadius: '50%', border: '3px solid rgba(255,255,255,0.15)',
                                borderTopColor: 'var(--pai-fresh)', animation: 'spin .7s linear infinite' }} />
                  <span style={{ fontSize: 11.5, color: 'rgba(255,255,255,0.55)', fontFamily: 'JetBrains Mono, monospace' }}>Loading…</span>
                </div>
              )}

              {imgSrc && !imgError && !noAnnotations && (
                // Keyed on tab/row/reloadKey (not the alpha-bearing imgSrc itself) so an
                // overlay-opacity change updates this element's src in place — the browser
                // keeps the previous frame visible until the new one decodes — rather than
                // remounting (and flashing blank) on every slider tick.
                <img key={`${tab}-${row._idx}-${reloadKey}`} src={imgSrc} alt={stem}
                     onLoad={() => setLoading(false)}
                     onError={async () => {
                       setLoading(false)
                       try {
                         const r = await fetch(imgSrc)
                         if (r.status === 404 && tab !== 'image') { setNoAnnotations(true); return }
                         const b = await r.json().catch(() => ({ error: `HTTP ${r.status}` }))
                         setImgError(b.error ?? `HTTP ${r.status}`)
                       } catch { setImgError('Could not reach the API.') }
                     }}
                     style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', borderRadius: 10, display: 'block',
                              boxShadow: '0 16px 50px rgba(0,0,0,0.55)', opacity: loading ? 0 : 1, transition: 'opacity .25s ease' }} />
              )}

              {!imgSrc && (
                <div style={{ textAlign: 'center', zIndex: 4 }}>
                  <p style={{ fontSize: 40, marginBottom: 10, color: 'rgba(255,255,255,0.18)' }}>◻</p>
                  <p style={{ fontSize: 13, color: 'rgba(255,255,255,0.6)', fontWeight: 600 }}>No image path detected</p>
                </div>
              )}
              {imgSrc && noAnnotations && (
                <div style={{ textAlign: 'center', zIndex: 4 }}>
                  <p style={{ fontSize: 40, marginBottom: 10, color: 'rgba(255,255,255,0.18)' }}>◻</p>
                  <p style={{ fontSize: 13, color: 'rgba(255,255,255,0.6)', fontWeight: 600 }}>No annotations for this image</p>
                  <p style={{ fontSize: 11.5, color: 'rgba(255,255,255,0.4)', marginTop: 4 }}>
                    This dataset has no COCO label file for it.
                  </p>
                </div>
              )}
              {imgSrc && imgError && !noAnnotations && (
                <div style={{ textAlign: 'center', zIndex: 4, maxWidth: 420 }}>
                  <div style={{ width: 52, height: 52, borderRadius: 14, margin: '0 auto 14px', display: 'flex',
                                alignItems: 'center', justifyContent: 'center', background: 'rgba(239,68,68,0.14)',
                                border: '1px solid rgba(239,68,68,0.4)' }}>
                    <AlertTriangle size={24} color="#fca5a5" />
                  </div>
                  <p style={{ fontSize: 13.5, fontWeight: 600, color: '#fff', marginBottom: 6 }}>Couldn’t load the image</p>
                  <p style={{ fontSize: 12, color: '#fca5a5', marginBottom: 16 }}>{imgError}</p>
                  <button onClick={() => setReloadKey(k => k + 1)}
                          style={{ display: 'inline-flex', alignItems: 'center', gap: 7, padding: '8px 16px', borderRadius: 9,
                                   fontSize: 12.5, fontWeight: 600, cursor: 'pointer', color: '#fff',
                                   background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)' }}>
                    <RotateCw size={13} /> Retry
                  </button>
                </div>
              )}
            </div>

            {/* full-res link (image path intentionally not shown) */}
            {origUrl && (
              <div className="flex items-center justify-end px-4 shrink-0"
                   style={{ height: 34, background: 'var(--pai-navy)', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                <a href={origUrl} target="_blank" rel="noreferrer"
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 11, fontWeight: 600,
                            color: 'var(--pai-fresh)', textDecoration: 'none', whiteSpace: 'nowrap' }}>
                  <ExternalLink size={12} /> Full resolution
                </a>
              </div>
            )}
          </div>

          {/* ── Dashboard summary ─────────────────────────────────────────────── */}
          <div className="flex flex-col shrink-0 overflow-y-auto"
               style={{ width: 372, borderLeft: '1px solid var(--c-border)', background: 'var(--c-surface)' }}>

            <div className="px-4 pt-4 pb-1">
              <p className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: 'var(--c-t3)' }}>Summary</p>
            </div>

            {/* Highlight tiles */}
            <div className="px-4 pt-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {heroes.map(c => (
                <StatCard key={c} label={prettify(c)} value={fmtVal(row[c])} />
              ))}
            </div>

            {/* Category / capture chips */}
            {(data.categoricalCols.length > 0 || partEntry?.cluster_id != null) && (
              <div className="px-4 pt-4">
                <p className="text-[10.5px] font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--c-t3)' }}>Attributes</p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {partEntry?.cluster_id != null && (
                    <Chip label="cluster" value={String(partEntry.cluster_id)} accent />
                  )}
                  {data.categoricalCols.map(c => (
                    <Chip key={c} label={prettify(c)} value={String(row[c] ?? '—')} />
                  ))}
                </div>
              </div>
            )}

            {/* Metrics with range bars */}
            {metricCols.length > 0 && (
              <div className="px-4 pt-4 pb-4">
                <p className="text-[10.5px] font-semibold uppercase tracking-wider mb-1.5" style={{ color: 'var(--c-t3)' }}>Metrics</p>
                <div>
                  {metricCols.map(c => (
                    <MetricBar key={c} label={prettify(c)} value={row[c]} meta={metaByName.get(c)} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ borderRadius: 11, padding: '10px 12px', background: 'var(--c-raised)', border: '1px solid var(--c-border)' }}>
      <div style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase',
                    color: 'var(--c-t3)', marginBottom: 4, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={label}>
        {label}
      </div>
      <div style={{ fontSize: 19, fontWeight: 700, color: 'var(--pai-navy)', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.1 }}>
        {value}
      </div>
    </div>
  )
}

function Chip({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 5, padding: '4px 9px', borderRadius: 7, fontSize: 11,
                   background: accent ? 'rgba(41,204,165,0.13)' : 'var(--c-raised)',
                   border: `1px solid ${accent ? 'rgba(41,204,165,0.4)' : 'var(--c-border)'}` }}>
      <span style={{ color: 'var(--c-t3)', fontSize: 9.5, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</span>
      <span style={{ color: accent ? 'var(--c-accent)' : 'var(--c-t1)', fontWeight: 600 }}>{value}</span>
    </span>
  )
}

function MetricBar({ label, value, meta }: { label: string; value: unknown; meta?: ColMeta }) {
  const v = Number(value)
  const hasRange = meta && Number.isFinite(meta.min) && Number.isFinite(meta.max) && (meta.max as number) > (meta.min as number)
  const pct = hasRange && Number.isFinite(v)
    ? Math.max(0, Math.min(1, (v - (meta!.min as number)) / ((meta!.max as number) - (meta!.min as number)))) : 0
  const medPct = hasRange && Number.isFinite(meta!.p50 as number)
    ? Math.max(0, Math.min(1, ((meta!.p50 as number) - (meta!.min as number)) / ((meta!.max as number) - (meta!.min as number)))) : null

  return (
    <div style={{ padding: '6px 0' }}>
      <div className="flex items-baseline justify-between" style={{ marginBottom: 4 }}>
        <span className="truncate" style={{ fontSize: 11.5, color: 'var(--c-t2)', maxWidth: 200 }} title={label}>{label}</span>
        <span style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--c-t1)', fontFamily: 'JetBrains Mono, monospace' }}>{fmtVal(value)}</span>
      </div>
      <div style={{ position: 'relative', height: 5, borderRadius: 3, background: 'var(--c-border)', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, width: `${pct * 100}%`, borderRadius: 3,
                      background: 'linear-gradient(90deg, var(--pai-teal), var(--pai-green))' }} />
        {medPct != null && (
          <div title="dataset median" style={{ position: 'absolute', top: -1, bottom: -1, left: `${medPct * 100}%`,
                       width: 1.5, background: 'rgba(1,55,85,0.45)' }} />
        )}
      </div>
    </div>
  )
}

function fmtVal(v: unknown): string {
  if (v == null || v === '') return '—'
  const n = Number(v)
  if (typeof v === 'number' || (v !== '' && Number.isFinite(n) && /^-?\d/.test(String(v)))) return fmtN(n)
  return String(v)
}

function fmtN(n: number) {
  if (n == null || !isFinite(n)) return '—'
  if (Number.isInteger(n)) return n.toLocaleString()
  if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(2)
  return n.toPrecision(4).replace(/\.?0+$/, '')
}
