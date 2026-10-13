import { useCallback, useEffect, useMemo, useRef } from 'react'
import type { KeyboardEvent } from 'react'
import { Cloud, Database, FlaskConical, GitBranch, Trash2 } from 'lucide-react'
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  type Edge,
  type FitViewOptions,
  type Node,
  type NodeProps,
  type ReactFlowInstance,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import './DatasetGraph.css'
import type { DatasetMeta, ExperimentCounts } from '../api'
import { experimentBadge } from '../lib/experimentBadge'
import { adjacentDataset, layoutDatasetGraph, lineageOf } from '../lib/datasetGraph'
import type { GraphDirection } from '../lib/datasetGraph'

interface Props {
  datasets: DatasetMeta[]
  /** One dataset selects it and shows its lineage; several mark a multi-selection (to merge). */
  selection: string[]
  experiments: ExperimentCounts
  /** `additive` is true for Shift/Ctrl/Cmd clicks; `null` clears the selection. */
  onSelect: (source: string | null, additive: boolean) => void
  onOpen: (source: string) => void
  onDelete?: (id: number) => void
}

type Emphasis = 'selected' | 'related' | 'dimmed' | 'idle'

interface DatasetNodeData extends Record<string, unknown> {
  dataset: DatasetMeta
  datasets: DatasetMeta[]
  emphasis: Emphasis
  experiments: ExperimentCounts[string] | undefined
  onDelete?: (id: number) => void
}

type DatasetFlowNode = Node<DatasetNodeData, 'dataset'>

const NODE_WIDTH = 210
const NODE_HEIGHT = 92
const FRAME_PADDING = 0.25
// Bottom padding keeps the fitted graph clear of the minimap.
const FIT_VIEW: FitViewOptions = { padding: { top: '56px', right: '40px', bottom: '130px', left: '40px' }, maxZoom: 1.1 }
const ARROW_DIRECTIONS: Record<string, GraphDirection> = {
  ArrowLeft: 'parent', ArrowRight: 'child', ArrowUp: 'up', ArrowDown: 'down',
}

function label(source: string, datasets: DatasetMeta[]): string {
  return datasets.find(dataset => dataset.source === source)?.name ?? source.split('/').pop() ?? source
}

function DatasetNodeCard({ data }: NodeProps<DatasetFlowNode>) {
  const { dataset, datasets, emphasis, experiments, onDelete } = data
  const badge = experimentBadge(experiments)
  const isCloud = dataset.name === 'Cloud Dataset'
  const parents = dataset.parent_sources ?? (dataset.parent_source ? [dataset.parent_source] : [])

  return (
    <article
      className={`dataset-graph__node dataset-graph__node--${emphasis}${isCloud ? ' dataset-graph__node--cloud' : ''}`}
    >
      <Handle className="dataset-graph__handle" type="target" position={Position.Left} />
      {dataset.deletable && dataset.id != null && onDelete && (
        <button
          className="dataset-graph__delete nodrag nopan"
          title="Delete dataset"
          aria-label={`Delete ${dataset.name}`}
          onClick={event => {
            event.stopPropagation()
            if (window.confirm(`Delete dataset "${dataset.name}"? This cannot be undone.`)) onDelete(dataset.id!)
          }}
        >
          <Trash2 size={11} />
        </button>
      )}
      <div className="dataset-graph__open">
        <span className="dataset-graph__icon">{isCloud ? <Cloud size={17} /> : <Database size={17} />}</span>
        <span className="dataset-graph__copy">
          <strong>{dataset.name}</strong>
          <small>{dataset.row_count?.toLocaleString() ?? 'CSV'} samples</small>
        </span>
        <span className={`dataset-graph__badge dataset-graph__badge--${badge.state}`} title={badge.label} role="img" aria-label={badge.label}>
          <FlaskConical size={13} />
        </span>
        {parents.length > 0 && (
          <span className="dataset-graph__parents" title={parents.map(parent => label(parent, datasets)).join(', ')}>
            <GitBranch size={10} /> {parents.length === 1 ? label(parents[0], datasets) : `${parents.length} parents`}
          </span>
        )}
      </div>
      <Handle className="dataset-graph__handle" type="source" position={Position.Right} />
    </article>
  )
}

const nodeTypes = { dataset: DatasetNodeCard }

const prefersReducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false

export function DatasetGraph({ datasets, selection, experiments, onSelect, onOpen, onDelete }: Props) {
  const selected = selection.length === 1 ? selection[0] : null
  const layout = useMemo(() => layoutDatasetGraph(datasets), [datasets])
  const lineage = useMemo(() => (selected ? lineageOf(layout.edges, selected) : null), [layout.edges, selected])
  const flow = useRef<ReactFlowInstance<DatasetFlowNode, Edge> | null>(null)

  const emphasisOf = useCallback((source: string): Emphasis => {
    if (selection.length > 1) return selection.includes(source) ? 'selected' : 'idle'
    if (!selected || !lineage) return 'idle'
    if (source === selected) return 'selected'
    return lineage.ancestors.has(source) || lineage.descendants.has(source) ? 'related' : 'dimmed'
  }, [lineage, selected, selection])

  const nodes = useMemo<DatasetFlowNode[]>(() => layout.nodes.map(({ dataset, x, y }) => ({
    id: dataset.source,
    type: 'dataset',
    position: { x, y },
    initialWidth: NODE_WIDTH,
    initialHeight: NODE_HEIGHT,
    data: { dataset, datasets, emphasis: emphasisOf(dataset.source), experiments: experiments[dataset.source], onDelete },
    draggable: false,
    selected: selection.includes(dataset.source),
  })), [datasets, emphasisOf, experiments, layout.nodes, onDelete, selection])

  const edges = useMemo<Edge[]>(() => layout.edges.map(edge => {
    const inLineage = !!selected && !!lineage && [edge.parent, edge.child].every(
      source => source === selected || lineage.ancestors.has(source) || lineage.descendants.has(source),
    )
    return {
      id: `${edge.parent}\u0000${edge.child}`,
      source: edge.parent,
      target: edge.child,
      type: 'smoothstep',
      animated: inLineage && !prefersReducedMotion(),
      className: selected ? (inLineage ? 'dataset-graph__edge--lineage' : 'dataset-graph__edge--dimmed') : undefined,
      markerEnd: { type: MarkerType.ArrowClosed },
    }
  }), [layout.edges, lineage, selected])

  const framed = useMemo(() => {
    if (selection.length > 1) return selection
    if (!selected || !lineage) return []
    return [selected, ...lineage.ancestors, ...lineage.descendants]
  }, [lineage, selected, selection])

  useEffect(() => {
    if (!framed.length) return
    // Wait for the detail panel's width transition so the framing is computed on the final viewport size.
    const timer = window.setTimeout(() => {
      void flow.current?.fitView({
        nodes: framed.map(id => ({ id })),
        padding: FRAME_PADDING,
        maxZoom: 1,
        duration: prefersReducedMotion() ? 0 : 350,
      })
    }, prefersReducedMotion() ? 0 : 320)
    return () => window.clearTimeout(timer)
  }, [framed])

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') return onSelect(null, false)
    if (event.key === 'Enter' && selected) return onOpen(selected)
    const direction = ARROW_DIRECTIONS[event.key]
    if (!direction) return
    event.preventDefault()
    const next = selected
      ? adjacentDataset(layout, selected, direction)
      : layout.nodes[0]?.dataset.source ?? null
    if (next) onSelect(next, false)
  }

  return (
    <div
      className="dataset-graph"
      role="group"
      tabIndex={0}
      aria-label="Interactive dataset lineage graph. Arrow keys move between datasets, Enter opens the explorer, Escape clears the selection."
      onKeyDown={onKeyDown}
    >
      <ReactFlow<DatasetFlowNode, Edge>
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onInit={instance => { flow.current = instance }}
        onNodeClick={(event, node) => onSelect(node.id, event.shiftKey || event.ctrlKey || event.metaKey)}
        onNodeDoubleClick={(_, node) => onOpen(node.id)}
        onPaneClick={() => onSelect(null, false)}
        fitView
        fitViewOptions={FIT_VIEW}
        minZoom={0.15}
        maxZoom={2}
        nodesDraggable={false}
        nodesConnectable={false}
        nodesFocusable={false}
        edgesFocusable={false}
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} />
        <Controls showInteractive={false} />
        <MiniMap<DatasetFlowNode>
          pannable
          zoomable
          nodeClassName={node => selection.includes(node.data.dataset.source) || node.data.dataset.name === 'Cloud Dataset'
            ? 'dataset-graph__mini dataset-graph__mini--accent'
            : 'dataset-graph__mini'}
          maskColor="var(--dataset-minimap-mask)"
          ariaLabel="Dataset lineage minimap"
        />
        <Panel position="top-right" className="dataset-graph__hint">
          Click to select · Shift-click to select several · double-click to explore · drag to pan · scroll to zoom
        </Panel>
      </ReactFlow>
    </div>
  )
}
