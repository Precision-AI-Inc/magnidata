import { useState, useMemo, useCallback } from 'react'
import { Play, ChevronUp, ChevronDown, ChevronsUpDown, ChevronLeft, ChevronRight } from 'lucide-react'
import type { CSVData, CSVRow } from '../types'
import { BRAND_CAT_BADGES, TABLE_HEADER, TABLE_HEADER_ACTIVE } from '../data/vizColors'

const PAGE_SIZE = 100

interface Props {
  data: CSVData
  rows: CSVRow[]
  onPlay: (row: CSVRow) => void
}

type SortDir = 'asc' | 'desc' | null

export function DataTable({ data, rows, onPlay }: Props) {
  const [sortCol, setSortCol]   = useState<string | null>(null)
  const [sortDir, setSortDir]   = useState<SortDir>(null)
  const [page, setPage]         = useState(0)
  const [colSearch, setColSearch] = useState('')

  // Visible columns
  const allCols  = data.cols.map(c => c.name)
  const visCols  = useMemo(() =>
    colSearch ? allCols.filter(c => c.toLowerCase().includes(colSearch.toLowerCase())) : allCols,
  [allCols, colSearch])

  // Sort
  const sorted = useMemo(() => {
    if (!sortCol || !sortDir) return rows
    return [...rows].sort((a, b) => {
      const va = a[sortCol], vb = b[sortCol]
      if (va == null && vb == null) return 0
      if (va == null) return 1
      if (vb == null) return -1
      if (typeof va === 'number' && typeof vb === 'number')
        return sortDir === 'asc' ? va - vb : vb - va
      const sa = String(va), sb = String(vb)
      return sortDir === 'asc' ? sa.localeCompare(sb) : sb.localeCompare(sa)
    })
  }, [rows, sortCol, sortDir])

  const totalPages = Math.ceil(sorted.length / PAGE_SIZE)
  const pageRows   = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const handleSort = useCallback((col: string) => {
    setSortCol(prev => {
      if (prev !== col) { setSortDir('desc'); return col }
      setSortDir(d => d === 'desc' ? 'asc' : d === 'asc' ? null : 'desc')
      return col
    })
    setPage(0)
  }, [])

  const hasImages = !!data.imagePath

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2.5 shrink-0"
           style={{ borderBottom: '1px solid var(--c-border)' }}>
        <span className="text-xs" style={{ color: 'var(--c-t3)' }}>
          {rows.length.toLocaleString()} rows · {data.cols.length} columns
        </span>
        <input
          type="text" placeholder="Filter columns…"
          value={colSearch} onChange={e => setColSearch(e.target.value)}
          className="ml-auto px-3 py-1 rounded-lg text-xs outline-none"
          style={{ background: 'var(--c-raised)', border: '1px solid var(--c-border)', color: 'var(--c-t1)', width: 180 }}
        />
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table style={{ borderCollapse: 'collapse', width: 'max-content', minWidth: '100%' }}>
          <thead>
            <tr style={{ position: 'sticky', top: 0, zIndex: 5 }}>
              {hasImages && (
                <th style={{ ...thStyle, width: 44, background: TABLE_HEADER }} />
              )}
              <th style={{ ...thStyle, width: 52, background: TABLE_HEADER, color: 'rgba(255,255,255,0.4)', fontSize: 10 }}>
                #
              </th>
              {visCols.map(col => (
                <SortableTh key={col} col={col}
                  active={sortCol === col} dir={sortCol === col ? sortDir : null}
                  onClick={() => handleSort(col)} />
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, ri) => (
              <tr key={row._idx as number}
                  className="group"
                  style={{ borderBottom: '1px solid var(--c-border)' }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'var(--c-hover)')}
                  onMouseLeave={e => (e.currentTarget.style.background = '')}>
                {hasImages && (
                  <td style={{ ...tdStyle, width: 44 }}>
                    <button className="play-btn" onClick={() => onPlay(row)} title="View image">
                      <Play size={10} fill="currentColor" />
                    </button>
                  </td>
                )}
                <td style={{ ...tdStyle, width: 52, color: 'var(--c-t3)', fontSize: 10, textAlign: 'right' }}>
                  {page * PAGE_SIZE + ri + 1}
                </td>
                {visCols.map(col => (
                  <DataCell key={col} value={row[col]} colName={col} data={data} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-2.5 shrink-0"
             style={{ borderTop: '1px solid var(--c-border)', background: 'var(--c-raised)' }}>
          <span className="text-xs" style={{ color: 'var(--c-t3)' }}>
            Page {page + 1} of {totalPages} ({sorted.length.toLocaleString()} rows)
          </span>
          <div className="flex items-center gap-1">
            <button className="btn-icon" disabled={page === 0} onClick={() => setPage(0)}>
              <ChevronLeft size={13} /><ChevronLeft size={13} style={{ marginLeft: -8 }} />
            </button>
            <button className="btn-icon" disabled={page === 0} onClick={() => setPage(p => p - 1)}>
              <ChevronLeft size={14} />
            </button>
            {pageRange(page, totalPages).map(p => (
              <button key={p}
                onClick={() => setPage(p)}
                className="w-7 h-7 rounded text-xs font-medium"
                style={{
                  background: p === page ? 'var(--c-accent)' : 'transparent',
                  color: p === page ? '#fff' : 'var(--c-t2)',
                }}>
                {p + 1}
              </button>
            ))}
            <button className="btn-icon" disabled={page === totalPages - 1} onClick={() => setPage(p => p + 1)}>
              <ChevronRight size={14} />
            </button>
            <button className="btn-icon" disabled={page === totalPages - 1} onClick={() => setPage(totalPages - 1)}>
              <ChevronRight size={13} /><ChevronRight size={13} style={{ marginLeft: -8 }} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function SortableTh({ col, active, dir, onClick }:
  { col: string; active: boolean; dir: SortDir; onClick: () => void }) {
  return (
    <th onClick={onClick} style={{
      ...thStyle,
      background: active ? TABLE_HEADER_ACTIVE : TABLE_HEADER,
      color: active ? '#fff' : 'rgba(255,255,255,0.72)',
      cursor: 'pointer',
      userSelect: 'none',
    }}>
      <div className="flex items-center gap-1">
        <span className="truncate" style={{ maxWidth: 140 }} title={col}>{col}</span>
        <span style={{ opacity: active ? 1 : 0.3, flexShrink: 0 }}>
          {!dir || !active ? <ChevronsUpDown size={10} /> :
           dir === 'asc'  ? <ChevronUp size={10} />     : <ChevronDown size={10} />}
        </span>
      </div>
    </th>
  )
}

function DataCell({ value, colName, data }:
  { value: unknown; colName: string; data: CSVData }) {
  const isNum = typeof value === 'number'
  const isCat = data.categoricalCols.includes(colName) && colName === 'category'

  if (value == null || value === '') {
    return <td style={{ ...tdStyle, color: 'var(--c-border-2)' }}>—</td>
  }

  if (isCat) {
    return (
      <td style={tdStyle}>
        <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold"
              style={{ background: catBg(Number(value)), color: catFg(Number(value)) }}>
          {String(value)}
        </span>
      </td>
    )
  }

  if (isNum) {
    const n = value as number
    return (
      <td style={{ ...tdStyle, textAlign: 'right', fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>
        {Number.isInteger(n) ? n.toLocaleString() : n.toPrecision(5).replace(/\.?0+$/, '')}
      </td>
    )
  }

  const str = String(value)
  const isPath = str.startsWith('/')
  return (
    <td style={{ ...tdStyle, maxWidth: 180 }}>
      <span className="block truncate" title={str} style={{ color: isPath ? 'var(--c-t3)' : 'var(--c-t1)', fontFamily: isPath ? 'monospace' : undefined, fontSize: isPath ? 10 : 12 }}>
        {isPath ? str.split('/').pop() : str}
      </span>
    </td>
  )
}

const thStyle: React.CSSProperties = {
  padding: '8px 10px', whiteSpace: 'nowrap', fontSize: 11, fontWeight: 500,
  borderBottom: '1px solid rgba(255,255,255,0.06)', textAlign: 'left',
  letterSpacing: '.3px',
}
const tdStyle: React.CSSProperties = {
  padding: '5px 10px', verticalAlign: 'middle', whiteSpace: 'nowrap', fontSize: 12,
}

function pageRange(current: number, total: number): number[] {
  const half = 2, start = Math.max(0, current - half), end = Math.min(total - 1, current + half)
  return Array.from({ length: end - start + 1 }, (_, i) => start + i)
}

function catBg(n: number) { return BRAND_CAT_BADGES[n % BRAND_CAT_BADGES.length][0] }
function catFg(n: number) { return BRAND_CAT_BADGES[n % BRAND_CAT_BADGES.length][1] }
