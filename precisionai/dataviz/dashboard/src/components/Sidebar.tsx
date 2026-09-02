import { useState } from 'react'
import { RotateCcw, ChevronDown, ChevronRight, SlidersHorizontal } from 'lucide-react'
import type { CSVData, FilterState, ColMeta } from '../types'
import { ColTooltip } from './ColTooltip'

interface Props {
  data: CSVData
  filters: FilterState
  activeCount: number
  onRange:    (col: string, r: [number, number]) => void
  onCategory: (col: string, v: Set<string>) => void
  onSearch:   (s: string) => void
  onReset:    () => void
}

export function Sidebar({ data, filters, activeCount, onRange, onCategory, onSearch, onReset }: Props) {
  return (
    <aside className="flex flex-col h-full shrink-0"
           style={{ width: 240, background: 'var(--c-surface)', borderRight: '1px solid var(--c-border)' }}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3"
           style={{ borderBottom: '1px solid var(--c-border)' }}>
        <div className="flex items-center gap-2">
          <SlidersHorizontal size={14} style={{ color: 'var(--c-accent)' }} />
          <span className="font-semibold text-xs uppercase tracking-wider" style={{ color: 'var(--c-t2)' }}>
            Filters
          </span>
          {activeCount > 0 && (
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full"
                  style={{ background: 'var(--c-accent)', color: '#fff' }}>
              {activeCount}
            </span>
          )}
        </div>
        {activeCount > 0 && (
          <button onClick={onReset} className="btn-icon" title="Reset all filters">
            <RotateCcw size={13} />
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2 flex flex-col gap-1">
        {/* Search */}
        <div className="pb-2 mb-1" style={{ borderBottom: '1px solid var(--c-border)' }}>
          <input
            type="text"
            placeholder="Search all columns…"
            value={filters.search}
            onChange={e => onSearch(e.target.value)}
            className="w-full px-3 py-1.5 rounded-lg text-xs outline-none"
            style={{
              background: 'var(--c-raised)', border: '1px solid var(--c-border)',
              color: 'var(--c-t1)',
            }}
          />
        </div>

        {/* Numeric range sliders */}
        {data.numericCols.length > 0 && (
          <Section label="Numeric" defaultOpen>
            {data.numericCols.map(col => {
              const meta = data.cols.find(c => c.name === col)!
              return (
                <RangeFilter key={col} meta={meta}
                  current={filters.ranges[col] ?? [meta.min!, meta.max!]}
                  onChange={r => onRange(col, r)} />
              )
            })}
          </Section>
        )}

        {/* Categorical checkboxes */}
        {data.categoricalCols.length > 0 && (
          <Section label="Categorical">
            {data.categoricalCols.map(col => {
              const meta = data.cols.find(c => c.name === col)!
              return (
                <CategoryFilter key={col} meta={meta}
                  selected={filters.categories[col] ?? new Set()}
                  onChange={v => onCategory(col, v)} />
              )
            })}
          </Section>
        )}
      </div>
    </aside>
  )
}

// ── Section collapsible ────────────────────────────────────────────────────────
function Section({ label, children, defaultOpen = false }:
  { label: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div>
      <button className="w-full flex items-center gap-1.5 py-1.5 text-[11px] font-semibold uppercase tracking-widest"
              style={{ color: 'var(--c-t3)' }}
              onClick={() => setOpen(o => !o)}>
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        {label}
      </button>
      {open && <div className="flex flex-col gap-2 mb-1">{children}</div>}
    </div>
  )
}

// ── Range filter ───────────────────────────────────────────────────────────────
function RangeFilter({ meta, current, onChange }:
  { meta: ColMeta; current: [number, number]; onChange: (r: [number, number]) => void }) {
  const colMin = meta.min ?? 0
  const colMax = meta.max ?? 1
  const range  = colMax - colMin || 1
  const [lo, hi] = current

  const isActive = lo > colMin || hi < colMax

  return (
    <div className="px-1 py-1.5 rounded-lg" style={{ background: isActive ? 'var(--c-hover)' : undefined }}>
      <div className="flex items-center justify-between mb-1">
        <ColTooltip col={meta.name} style={{ maxWidth: 110 }}
                    className="text-[11px] font-medium truncate" >
          {meta.name}
        </ColTooltip>
        <span className="text-[10px] font-mono" style={{ color: 'var(--c-t3)' }}>
          {fmt(lo)} – {fmt(hi)}
        </span>
      </div>

      {/* Dual slider */}
      <div className="relative h-4 flex items-center">
        {/* Track fill */}
        <div className="absolute h-1 rounded-full w-full" style={{ background: 'var(--c-border-2)' }} />
        <div className="absolute h-1 rounded-full"
             style={{
               background: isActive ? 'var(--c-accent)' : 'var(--c-border-2)',
               left: `${((lo - colMin) / range) * 100}%`,
               right: `${((colMax - hi) / range) * 100}%`,
             }} />
        <input type="range" min={colMin} max={colMax} step={(range) / 200}
               value={lo}
               onChange={e => {
                 const v = Math.min(parseFloat(e.target.value), hi)
                 onChange([v, hi])
               }}
               className="absolute w-full opacity-0 cursor-pointer h-4" style={{ zIndex: 2 }} />
        <input type="range" min={colMin} max={colMax} step={(range) / 200}
               value={hi}
               onChange={e => {
                 const v = Math.max(parseFloat(e.target.value), lo)
                 onChange([lo, v])
               }}
               className="absolute w-full opacity-0 cursor-pointer h-4" style={{ zIndex: 3 }} />
        {/* Thumbs */}
        <div className="absolute w-3 h-3 rounded-full border-2 pointer-events-none"
             style={{
               left: `calc(${((lo - colMin) / range) * 100}% - 6px)`,
               background: 'var(--c-surface)', borderColor: isActive ? 'var(--c-accent)' : 'var(--c-border-2)',
             }} />
        <div className="absolute w-3 h-3 rounded-full border-2 pointer-events-none"
             style={{
               left: `calc(${((hi - colMin) / range) * 100}% - 6px)`,
               background: 'var(--c-surface)', borderColor: isActive ? 'var(--c-accent)' : 'var(--c-border-2)',
             }} />
      </div>
    </div>
  )
}

// ── Category filter ────────────────────────────────────────────────────────────
function CategoryFilter({ meta, selected, onChange }:
  { meta: ColMeta; selected: Set<string>; onChange: (v: Set<string>) => void }) {
  const [expanded, setExpanded] = useState(false)
  const items = meta.topValues ?? []
  const shown = expanded ? items : items.slice(0, 6)
  const isActive = selected.size > 0

  const toggle = (v: string) => {
    const next = new Set(selected)
    next.has(v) ? next.delete(v) : next.add(v)
    onChange(next)
  }

  return (
    <div className="px-1 py-1.5 rounded-lg" style={{ background: isActive ? 'var(--c-hover)' : undefined }}>
      <div className="flex items-center justify-between mb-1.5">
        <ColTooltip col={meta.name} className="text-[11px] font-medium" style={{ color: 'var(--c-t2)' }}>
          {meta.name}
        </ColTooltip>
        {isActive && (
          <button onClick={() => onChange(new Set())} className="text-[10px]"
                  style={{ color: 'var(--c-accent)' }}>clear</button>
        )}
      </div>
      <div className="flex flex-col gap-0.5">
        {shown.map(([val, count]) => (
          <label key={val} className="flex items-center gap-1.5 cursor-pointer py-0.5 px-1 rounded"
                 style={{ background: selected.has(val) ? 'var(--c-active-row)' : undefined }}>
            <input type="checkbox" checked={selected.has(val)} onChange={() => toggle(val)}
                   className="w-3 h-3 rounded" style={{ accentColor: 'var(--c-accent)' }} />
            <span className="flex-1 text-[11px] truncate" style={{ color: 'var(--c-t1)' }}>{val}</span>
            <span className="text-[10px] shrink-0" style={{ color: 'var(--c-t3)' }}>{count}</span>
          </label>
        ))}
      </div>
      {items.length > 6 && (
        <button onClick={() => setExpanded(e => !e)} className="text-[10px] mt-1"
                style={{ color: 'var(--c-accent)' }}>
          {expanded ? 'Show less' : `+${items.length - 6} more`}
        </button>
      )}
    </div>
  )
}

function fmt(n: number) {
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (Math.abs(n) < 0.01 && n !== 0) return n.toExponential(2)
  return n.toPrecision(4).replace(/\.?0+$/, '')
}
