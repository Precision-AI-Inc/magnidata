import { useMemo } from 'react'
import { Cloud, Database, GitBranch, Loader2, Trash2 } from 'lucide-react'
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
  type Node,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { DatasetMeta } from '../api'
import { layoutDatasetGraph } from '../lib/datasetGraph'

interface Props {
  datasets: DatasetMeta[]
  loadingSource: string | null
  onOpen: (dataset: DatasetMeta) => void
  onDelete?: (id: number) => void
}

interface DatasetNodeData extends Record<string, unknown> {
  dataset: DatasetMeta
  datasets: DatasetMeta[]
  loadingSource: string | null
  onOpen: (dataset: DatasetMeta) => void
  onDelete?: (id: number) => void
}

type DatasetFlowNode = Node<DatasetNodeData, 'dataset'>

function label(source: string, datasets: DatasetMeta[]): string {
  return datasets.find(dataset => dataset.source === source)?.name ?? source.split('/').pop() ?? source
}

function DatasetNodeCard({ data }: NodeProps<DatasetFlowNode>) {
  const { dataset, datasets, loadingSource, onOpen, onDelete } = data
  const active = loadingSource === dataset.source
  const isCloud = dataset.name === 'Cloud Dataset'
  const parents = dataset.parent_sources ?? (dataset.parent_source ? [dataset.parent_source] : [])

  return (
    <article className={`dataset-graph__node${isCloud ? ' dataset-graph__node--cloud' : ''}`}>
      <Handle className="dataset-graph__handle" type="target" position={Position.Left} />
      {dataset.deletable && dataset.id != null && onDelete && (
        <button
          className="dataset-graph__delete nodrag nopan"
          title="Delete dataset"
          aria-label={`Delete ${dataset.name}`}
          onClick={() => {
            if (window.confirm(`Delete dataset "${dataset.name}"? This cannot be undone.`)) onDelete(dataset.id!)
          }}
        >
          <Trash2 size={11} />
        </button>
      )}
      <button
        className="dataset-graph__open nodrag nopan"
        onClick={() => onOpen(dataset)}
        disabled={loadingSource !== null}
      >
        <span className="dataset-graph__icon">
          {active ? <Loader2 size={17} className="spin" /> : isCloud ? <Cloud size={17} /> : <Database size={17} />}
        </span>
        <span className="dataset-graph__copy">
          <strong>{dataset.name}</strong>
          <small>{dataset.row_count?.toLocaleString() ?? 'CSV'} samples</small>
        </span>
        {parents.length > 0 && (
          <span className="dataset-graph__parents" title={parents.map(parent => label(parent, datasets)).join(', ')}>
            <GitBranch size={10} /> {parents.length === 1 ? label(parents[0], datasets) : `${parents.length} parents`}
          </span>
        )}
      </button>
      <Handle className="dataset-graph__handle" type="source" position={Position.Right} />
    </article>
  )
}

const nodeTypes = { dataset: DatasetNodeCard }

export function DatasetGraph({ datasets, loadingSource, onOpen, onDelete }: Props) {
  const layout = useMemo(() => layoutDatasetGraph(datasets), [datasets])
  const nodes = useMemo<DatasetFlowNode[]>(() => layout.nodes.map(({ dataset, x, y }) => ({
    id: dataset.source,
    type: 'dataset',
    position: { x, y },
    data: { dataset, datasets, loadingSource, onOpen, onDelete },
    draggable: false,
    selectable: false,
  })), [datasets, layout.nodes, loadingSource, onDelete, onOpen])
  const edges = useMemo<Edge[]>(() => layout.edges.map(edge => ({
    id: `${edge.parent}\u0000${edge.child}`,
    source: edge.parent,
    target: edge.child,
    type: 'smoothstep',
    markerEnd: { type: MarkerType.ArrowClosed },
  })), [layout.edges])

  return (
    <div className="dataset-graph" aria-label="Interactive dataset lineage graph">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2, maxZoom: 1.1 }}
        minZoom={0.15}
        maxZoom={2}
        nodesDraggable={false}
        nodesConnectable={false}
        nodesFocusable={false}
        edgesFocusable={false}
        elementsSelectable={false}
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} />
        <Controls showInteractive={false} />
        <MiniMap<DatasetFlowNode>
          pannable
          zoomable
          nodeColor={node => node.data.dataset.name === 'Cloud Dataset' ? 'var(--c-accent)' : 'var(--c-border-2)'}
          maskColor="var(--dataset-minimap-mask)"
          ariaLabel="Dataset lineage minimap"
        />
        <Panel position="top-right" className="dataset-graph__hint">Drag to pan · scroll to zoom</Panel>
      </ReactFlow>
    </div>
  )
}
