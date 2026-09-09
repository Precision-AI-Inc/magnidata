import { useState, useEffect, useCallback, useMemo } from 'react'
import { api } from '../api'
import type { CSVData, CSVRow } from '../types'

export interface PartitionEntry {
  cluster_id: number | null   // 1-based; null = unassigned
}

/** The image filename used as the persistence/join key for a row (matches the backend & embeddings). */
export function imageNameOf(row: CSVRow, data: CSVData): string | null {
  if (!data.imagePath) return null
  const v = String(row[data.imagePath] ?? '')
  if (!v) return null
  return v.split(/[/\\]/).pop() || null
}

export interface Partitions {
  map: Map<string, PartitionEntry>
  loading: boolean
  entryOf: (row: CSVRow) => PartitionEntry | undefined
  /** Commit a clustering — assign cluster ids to the given images. */
  commit: (entries: { image_name: string; cluster_id: number }[]) => Promise<number>
  stats: { committed: number }
}

/**
 * Persistent cluster-partition annotations (cluster_id) for the active dataset.
 * Loads from the API when the dataset changes; mutations write through to the API and update
 * the in-memory map so the 3D view, table, and preview stay in sync. Keyed by image filename.
 */
export function usePartitions(
  datasetSource: string | null | undefined,
  data: CSVData | null,
): Partitions {
  const [map, setMap] = useState<Map<string, PartitionEntry>>(new Map())
  const [loading, setLoading] = useState(false)

  // (Re)load whenever the dataset changes.
  useEffect(() => {
    setMap(new Map())
    if (!datasetSource) return
    let alive = true
    setLoading(true)
    api.getPartitions(datasetSource)
      .then(r => {
        if (!alive) return
        setMap(new Map(r.partitions.map(p => [p.image_name, { cluster_id: p.cluster_id }])))
      })
      .catch(() => { if (alive) setMap(new Map()) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [datasetSource])

  const entryOf = useCallback((row: CSVRow): PartitionEntry | undefined => {
    if (!data) return undefined
    const name = imageNameOf(row, data)
    return name ? map.get(name) : undefined
  }, [data, map])

  const commit = useCallback(async (entries: { image_name: string; cluster_id: number }[]) => {
    if (!datasetSource || !entries.length) return 0
    await api.commitPartitions(datasetSource, entries)
    setMap(prev => {
      const next = new Map(prev)
      for (const e of entries) next.set(e.image_name, { cluster_id: e.cluster_id })
      return next
    })
    return entries.length
  }, [datasetSource])

  const stats = useMemo(() => {
    let committed = 0
    for (const v of map.values()) if (v.cluster_id != null) committed++
    return { committed }
  }, [map])

  return { map, loading, entryOf, commit, stats }
}
