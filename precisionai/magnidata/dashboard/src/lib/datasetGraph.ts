import type { DatasetMeta } from '../api'

export interface DatasetGraphNode {
  dataset: DatasetMeta
  depth: number
  x: number
  y: number
}

export interface DatasetGraphEdge {
  parent: string
  child: string
}

export interface DatasetGraphLayout {
  nodes: DatasetGraphNode[]
  edges: DatasetGraphEdge[]
  height: number
}

/** Lay datasets out in lineage columns, retaining multi-parent edges. */
export function layoutDatasetGraph(datasets: DatasetMeta[]): DatasetGraphLayout {
  const sources = new Set(datasets.map(dataset => dataset.source))
  const parents = new Map(datasets.map(dataset => [
    dataset.source,
    (dataset.parent_sources ?? (dataset.parent_source ? [dataset.parent_source] : [])).filter(source => sources.has(source)),
  ]))
  const depths = new Map<string, number>()
  const depthOf = (source: string, visiting = new Set<string>()): number => {
    if (depths.has(source)) return depths.get(source)!
    if (visiting.has(source)) return 0
    const next = new Set(visiting).add(source)
    const lineage = parents.get(source) ?? []
    const depth = lineage.length ? Math.max(...lineage.map(parent => depthOf(parent, next))) + 1 : 0
    depths.set(source, depth)
    return depth
  }
  datasets.forEach(dataset => depthOf(dataset.source))
  const maxDepth = Math.max(0, ...depths.values())
  const levels = Array.from({ length: maxDepth + 1 }, () => [] as DatasetMeta[])
  datasets.forEach(dataset => levels[depths.get(dataset.source) ?? 0].push(dataset))
  levels.forEach(level => level.sort((a, b) => a.name.localeCompare(b.name)))
  const height = Math.max(340, Math.max(...levels.map(level => level.length), 1) * 116)
  const nodes = levels.flatMap((level, depth) => level.map((dataset, index) => ({
    dataset,
    depth,
    x: maxDepth ? 8 + (depth / maxDepth) * 84 : 50,
    y: ((index + 1) / (level.length + 1)) * 100,
  })))
  const edges = datasets.flatMap(dataset => (parents.get(dataset.source) ?? []).map(parent => ({
    parent,
    child: dataset.source,
  })))
  return { nodes, edges, height }
}
