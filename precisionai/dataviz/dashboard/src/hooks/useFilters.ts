import { useState, useCallback, useMemo } from 'react'
import type { CSVData, CSVRow, FilterState } from '../types'

const emptyFilters = (): FilterState => ({
  ranges: {}, categories: {}, search: '', pcConstraints: {},
})

export function useFilters(data: CSVData | null) {
  const [filters, setFilters] = useState<FilterState>(emptyFilters)

  const setRange = useCallback((col: string, range: [number, number]) => {
    setFilters(f => ({ ...f, ranges: { ...f.ranges, [col]: range } }))
  }, [])

  const setCategory = useCallback((col: string, values: Set<string>) => {
    setFilters(f => ({ ...f, categories: { ...f.categories, [col]: values } }))
  }, [])

  const setSearch = useCallback((s: string) => {
    setFilters(f => ({ ...f, search: s }))
  }, [])

  const setPCConstraints = useCallback((c: Record<string, [number, number][]>) => {
    setFilters(f => ({ ...f, pcConstraints: c }))
  }, [])

  const reset = useCallback(() => setFilters(emptyFilters()), [])

  const filteredRows = useMemo<CSVRow[]>(() => {
    if (!data) return []
    const { ranges, categories, search, pcConstraints } = filters

    const searchLower = search.toLowerCase()

    return data.rows.filter(row => {
      // Search
      if (searchLower) {
        const hit = data.cols.some(c => {
          const v = row[c.name]
          return v != null && String(v).toLowerCase().includes(searchLower)
        })
        if (!hit) return false
      }

      // Numeric range filters
      for (const [col, [lo, hi]] of Object.entries(ranges)) {
        const v = row[col] as number | undefined
        if (v == null) continue
        if (v < lo || v > hi) return false
      }

      // Category filters
      for (const [col, allowed] of Object.entries(categories)) {
        if (allowed.size === 0) continue
        const v = String(row[col] ?? '')
        if (!allowed.has(v)) return false
      }

      // Parallel-coords brush constraints
      for (const [col, ranges2] of Object.entries(pcConstraints)) {
        if (!ranges2.length) continue
        const enc = data.catEncodings[col]
        if (enc) {
          // Categorical axis: Plotly encodes values as integers; compare encoded index
          const encoded = enc.encode[String(row[col] ?? '')]
          if (encoded == null) return false
          if (!ranges2.some(([lo, hi]) => encoded >= lo && encoded <= hi)) return false
        } else {
          const v = row[col] as number | undefined
          if (v == null) continue
          if (!ranges2.some(([lo, hi]) => v >= lo && v <= hi)) return false
        }
      }

      return true
    })
  }, [data, filters])

  const activeCount = useMemo(() => {
    let n = 0
    n += Object.keys(filters.ranges).length
    n += Object.values(filters.categories).filter(s => s.size > 0).length
    n += filters.search ? 1 : 0
    n += Object.values(filters.pcConstraints).filter(r => r.length).length
    return n
  }, [filters])

  return { filters, filteredRows, setRange, setCategory, setSearch, setPCConstraints, reset, activeCount }
}
