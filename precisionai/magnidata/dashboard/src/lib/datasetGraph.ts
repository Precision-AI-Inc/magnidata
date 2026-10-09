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
  width: number
  height: number
}

const COLUMN_GAP = 290
const ROW_GAP = 132
const GRAPH_PADDING = 72
const NODE_WIDTH = 210
const NODE_HEIGHT = 92

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
  const largestLevel = Math.max(...levels.map(level => level.length), 1)
  const width = GRAPH_PADDING * 2 + maxDepth * COLUMN_GAP + NODE_WIDTH
  const height = GRAPH_PADDING * 2 + (largestLevel - 1) * ROW_GAP + NODE_HEIGHT
  const nodes = levels.flatMap((level, depth) => level.map((dataset, index) => ({
    dataset,
    depth,
    x: GRAPH_PADDING + depth * COLUMN_GAP,
    y: GRAPH_PADDING + index * ROW_GAP + ((largestLevel - level.length) * ROW_GAP) / 2,
  })))
  const edges = datasets.flatMap(dataset => (parents.get(dataset.source) ?? []).map(parent => ({
    parent,
    child: dataset.source,
  })))
  return { nodes, edges, width, height }
}
