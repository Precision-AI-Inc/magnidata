import { useState, useMemo, type CSSProperties } from 'react'
import { Search, X, Eye, EyeOff, ChevronDown, ChevronRight } from 'lucide-react'
import { COL_DESCS, GROUP_DESCS, GROUPS_ORDER, GROUP_COLORS } from '../data/colDescs'
import type { CSVData } from '../types'

interface Props {
  data: CSVData | null
  hiddenCols: Set<string>
  onToggleCol: (col: string) => void
}

export function Schema({ data, hiddenCols, onToggleCol }: Props) {
  const [search,       setSearch]       = useState('')
  const [activeGroup,  setActiveGroup]  = useState<string | null>(null)
  const [othersOpen,   setOthersOpen]   = useState(false)

  // Build a list of all known columns, marking which ones are present in the loaded CSV.
  // When a dataset is loaded, columns NOT present in it are reclassified into "Others" so
  // they can be tucked into a collapsible panel and not clutter the relevant groups.
  const allEntries = useMemo(() => {
    const loadedCols = new Set(data?.cols.map(c => c.name) ?? [])
    return Object.entries(COL_DESCS).map(([col, desc]) => {
      const inDataset = loadedCols.has(col)
      return { col, ...desc, inDataset, group: (data && !inDataset) ? 'Others' : desc.group }
    })
  }, [data])

  const filtered = useMemo(() => {
    const q = search.toLowerCase()
    return allEntries.filter(e => {
      if (activeGroup && e.group !== activeGroup) return false
      if (!q) return true
      return (
        e.col.toLowerCase().includes(q) ||
        e.plain_meaning.toLowerCase().includes(q) ||
        e.group.toLowerCase().includes(q)
      )
    })
  }, [allEntries, search, activeGroup])

  // Count per group (unfiltered by group, filtered by search)
  const groupCounts = useMemo(() => {
    const q = search.toLowerCase()
    const counts: Record<string, number> = {}
    allEntries.forEach(e => {
      if (q && !(e.col.toLowerCase().includes(q) || e.plain_meaning.toLowerCase().includes(q))) return
      counts[e.group] = (counts[e.group] ?? 0) + 1
    })
    return counts
  }, [allEntries, search])

  const totalShown = useMemo(() => {
    const q = search.toLowerCase()
    if (!q) return allEntries.length
    return allEntries.filter(e =>
      e.col.toLowerCase().includes(q) || e.plain_meaning.toLowerCase().includes(q)
    ).length
  }, [allEntries, search])

  // Split: "Others" (not-in-dataset) cards go into a collapsible panel in the All view.
  const showingOthers = activeGroup === 'Others'
  const otherEntries  = filtered.filter(e => e.group === 'Others')
  const gridEntries   = showingOthers ? otherEntries : filtered.filter(e => e.group !== 'Others')
  const showOthersPanel = !activeGroup && otherEntries.length > 0
  const othersExpanded  = othersOpen || !!search.trim()
  const CARD_GRID: CSSProperties = {
    display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 10,
  }
  const renderCard = (e: typeof allEntries[number]) => (
    <ColCard key={e.col} entry={e} isHidden={hiddenCols.has(e.col)} onToggle={() => onToggleCol(e.col)} />
  )

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden', background: 'var(--c-bg)' }}>

      {/* ── Left sidebar: group nav ── */}
      <div style={{
        width: 220, flexShrink: 0, overflowY: 'auto',
        borderRight: '1px solid var(--c-border)',
        background: 'var(--c-surface)',
        display: 'flex', flexDirection: 'column',
      }}>
        <div style={{ padding: '12px 12px 8px', borderBottom: '1px solid var(--c-border)' }}>
          <p style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
                      letterSpacing: '.6px', color: 'var(--c-t3)', marginBottom: 2 }}>
            Schema
          </p>
          <p style={{ fontSize: 11, color: 'var(--c-t3)', lineHeight: 1.4 }}>
            {data
              ? `${data.cols.length} of ${Object.keys(COL_DESCS).length} columns loaded`
              : `${Object.keys(COL_DESCS).length} columns documented`}
          </p>
        </div>

        <div style={{ padding: '8px 6px', flex: 1 }}>
          {/* All */}
          <GroupRow
            label="All columns"
            count={totalShown}
            color="#64748b"
            active={!activeGroup}
            onClick={() => setActiveGroup(null)}
          />

          <div style={{ height: 1, background: 'var(--c-border)', margin: '6px 4px' }} />

          {[...GROUPS_ORDER, 'Others'].map(g => {
            const count = groupCounts[g] ?? 0
            if (!count) return null
            return (
              <GroupRow
                key={g}
                label={g}
                count={count}
                color={GROUP_COLORS[g] ?? '#64748b'}
                active={activeGroup === g}
                onClick={() => setActiveGroup(g === activeGroup ? null : g)}
              />
            )
          })}
        </div>

        {/* About section */}
        <div style={{ padding: '10px 12px', borderTop: '1px solid var(--c-border)',
                      background: 'var(--c-raised)' }}>
          <p style={{ fontSize: 10, color: 'var(--c-t3)', lineHeight: 1.5 }}>
            Most measurements are <strong style={{ color: 'var(--c-t2)' }}>0–1 scores</strong>.
            Higher = more of that property. See scale for exceptions.
          </p>
        </div>
      </div>

      {/* ── Right: column cards ── */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>

        {/* Search bar */}
        <div style={{
          padding: '10px 16px', borderBottom: '1px solid var(--c-border)',
          background: 'var(--c-surface)', flexShrink: 0,
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          {activeGroup && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '3px 10px', borderRadius: 12,
              background: (GROUP_COLORS[activeGroup] ?? '#64748b') + '22',
              border: `1px solid ${(GROUP_COLORS[activeGroup] ?? '#64748b')}44`,
              fontSize: 11, fontWeight: 600, flexShrink: 0,
              color: GROUP_COLORS[activeGroup] === '#013755' ? '#006A7C' : GROUP_COLORS[activeGroup],
            }}>
              {activeGroup}
              <button onClick={() => setActiveGroup(null)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer',
                               padding: 0, color: 'inherit', opacity: 0.6, display: 'flex' }}>
                <X size={10} />
              </button>
            </div>
          )}

          <div style={{ position: 'relative', flex: 1, maxWidth: 360 }}>
            <Search size={13} style={{ position: 'absolute', left: 10, top: '50%',
                                        transform: 'translateY(-50%)', color: 'var(--c-t3)' }} />
            <input
              placeholder="Search columns or meanings…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{
                width: '100%', padding: '6px 10px 6px 30px', borderRadius: 8,
                fontSize: 12, outline: 'none',
                background: 'var(--c-raised)', border: '1px solid var(--c-border)',
                color: 'var(--c-t1)',
              }}
            />
            {search && (
              <button onClick={() => setSearch('')}
                      style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)',
                               background: 'none', border: 'none', cursor: 'pointer',
                               color: 'var(--c-t3)', padding: 0, display: 'flex' }}>
                <X size={12} />
              </button>
            )}
          </div>

          <span style={{ fontSize: 11, color: 'var(--c-t3)', flexShrink: 0 }}>
            {filtered.length} column{filtered.length !== 1 ? 's' : ''}
          </span>

          {hiddenCols.size > 0 && (
            <button
              onClick={() => [...hiddenCols].forEach(c => onToggleCol(c))}
              style={{
                display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0,
                padding: '3px 10px', borderRadius: 12, cursor: 'pointer',
                background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)',
                fontSize: 11, fontWeight: 600, color: '#ef4444',
              }}
            >
              <EyeOff size={10} />
              {hiddenCols.size} hidden · Show all
            </button>
          )}
        </div>

        {/* Cards grid */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 16px' }}>
          {filtered.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--c-t3)' }}>
              <p style={{ fontSize: 28, marginBottom: 8 }}>🔍</p>
              <p>No columns match your search</p>
            </div>
          ) : (
            <>
              {gridEntries.length > 0 && (
                <div style={CARD_GRID}>{gridEntries.map(renderCard)}</div>
              )}

              {/* Collapsible "Others" panel — documented columns not in this dataset */}
              {showOthersPanel && (
                <div style={{ marginTop: gridEntries.length > 0 ? 18 : 0, borderRadius: 10,
                              border: '1px solid var(--c-border)', background: 'var(--c-surface)', overflow: 'hidden' }}>
                  <button onClick={() => setOthersOpen(o => !o)}
                          style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 8, padding: '11px 14px',
                                   cursor: 'pointer', background: 'var(--c-raised)', border: 'none',
                                   borderBottom: othersExpanded ? '1px solid var(--c-border)' : 'none' }}>
                    {othersExpanded ? <ChevronDown size={15} style={{ color: 'var(--c-t3)' }} />
                                    : <ChevronRight size={15} style={{ color: 'var(--c-t3)' }} />}
                    <span style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--c-t1)' }}>Others</span>
                    <span style={{ fontSize: 11, color: 'var(--c-t3)' }}>documented, not in this dataset</span>
                    <span style={{ marginLeft: 'auto', fontSize: 11, fontWeight: 700, padding: '1px 8px',
                                   borderRadius: 10, background: 'var(--c-border)', color: 'var(--c-t2)' }}>
                      {otherEntries.length}
                    </span>
                  </button>
                  {othersExpanded && (
                    <div style={{ ...CARD_GRID, padding: 12 }}>{otherEntries.map(renderCard)}</div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Group nav row ──────────────────────────────────────────────────────────────
function GroupRow({ label, count, color, active, onClick }: {
  label: string; count: number; color: string; active: boolean; onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      style={{
        width: '100%', display: 'flex', alignItems: 'center', gap: 7,
        padding: '5px 8px', borderRadius: 7, cursor: 'pointer', textAlign: 'left',
        background: active ? color + '22' : 'transparent',
        border: active ? `1px solid ${color}44` : '1px solid transparent',
        marginBottom: 2,
      }}
    >
      <span style={{ width: 8, height: 8, borderRadius: 2, flexShrink: 0, background: color }} />
      <span style={{
        flex: 1, fontSize: 11, lineHeight: 1.3, overflow: 'hidden',
        textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        fontWeight: active ? 600 : 400,
        color: active ? 'var(--c-t1)' : 'var(--c-t2)',
      }}>
        {label}
      </span>
      <span style={{
        fontSize: 10, fontWeight: 600, minWidth: 20, textAlign: 'right', flexShrink: 0,
        color: active ? color : 'var(--c-t3)',
      }}>
        {count}
      </span>
    </button>
  )
}

// ── Column card ────────────────────────────────────────────────────────────────
type EntryItem = {
  col: string; group: string; plain_meaning: string; scale: string; inDataset: boolean
}

function ColCard({ entry, isHidden, onToggle }: {
  entry: EntryItem
  isHidden: boolean
  onToggle: () => void
}) {
  const color = GROUP_COLORS[entry.group] ?? '#64748b'
  return (
    <div style={{
      borderRadius: 10, padding: '12px 14px',
      background: 'var(--c-surface)', border: `1px solid ${isHidden ? 'rgba(239,68,68,0.2)' : 'var(--c-border)'}`,
      opacity: isHidden ? 0.5 : entry.inDataset ? 1 : 0.55,
      transition: 'box-shadow .15s, opacity .15s, border-color .15s',
    }}
    onMouseEnter={e => (e.currentTarget.style.boxShadow = `0 0 0 2px ${color}44`)}
    onMouseLeave={e => (e.currentTarget.style.boxShadow = '')}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 8 }}>
        <code style={{
          flex: 1, fontSize: 12, fontWeight: 700,
          fontFamily: 'JetBrains Mono, monospace',
          color: 'var(--c-t1)', wordBreak: 'break-all',
        }}>
          {entry.col}
        </code>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 3, flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <button
              onClick={onToggle}
              title={isHidden ? 'Show column' : 'Hide column from analysis'}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                width: 22, height: 22, borderRadius: 6, cursor: 'pointer',
                background: isHidden ? 'rgba(239,68,68,0.12)' : 'var(--c-raised)',
                border: `1px solid ${isHidden ? 'rgba(239,68,68,0.3)' : 'var(--c-border)'}`,
                color: isHidden ? '#ef4444' : 'var(--c-t3)',
                transition: 'all .15s',
                padding: 0,
              }}
            >
              {isHidden ? <EyeOff size={11} /> : <Eye size={11} />}
            </button>
            <span style={{
              fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.5px',
              padding: '2px 6px', borderRadius: 6,
              background: color + '22', color: color === '#013755' ? '#006A7C' : color,
              border: `1px solid ${color}33`,
            }}>
              {entry.group}
            </span>
          </div>
          {!entry.inDataset && (
            <span style={{ fontSize: 9, color: 'var(--c-t3)', fontStyle: 'italic' }}>
              not in file
            </span>
          )}
        </div>
      </div>

      {/* Plain meaning */}
      <p style={{ fontSize: 12, color: 'var(--c-t2)', lineHeight: 1.55, margin: 0 }}>
        {entry.plain_meaning}
      </p>

      {/* Scale */}
      <div style={{
        marginTop: 8, paddingTop: 7, borderTop: '1px solid var(--c-border)',
        fontSize: 10, color: 'var(--c-t3)', fontStyle: 'italic',
        display: 'flex', alignItems: 'center', gap: 4,
      }}>
        <span style={{ fontStyle: 'normal', fontWeight: 600, color: 'var(--c-t3)' }}>Scale:</span>
        {entry.scale}
      </div>
    </div>
  )
}

