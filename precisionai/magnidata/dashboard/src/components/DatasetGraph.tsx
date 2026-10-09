import { Cloud, Database, GitBranch, Loader2, Trash2 } from 'lucide-react'
import type { DatasetMeta } from '../api'
import { layoutDatasetGraph } from '../lib/datasetGraph'

interface Props {
  datasets: DatasetMeta[]
  loadingSource: string | null
  onOpen: (dataset: DatasetMeta) => void
  onDelete?: (id: number) => void
}

function label(source: string, datasets: DatasetMeta[]): string {
  return datasets.find(dataset => dataset.source === source)?.name ?? source.split('/').pop() ?? source
}

export function DatasetGraph({ datasets, loadingSource, onOpen, onDelete }: Props) {
  const layout = layoutDatasetGraph(datasets)
  const positions = new Map(layout.nodes.map(node => [node.dataset.source, node]))
  return (
    <div className="dataset-graph" style={{ height: layout.height }}>
      <svg className="dataset-graph__edges" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
        <defs>
          <marker id="dataset-arrow" markerWidth="4" markerHeight="4" refX="3" refY="2" orient="auto">
            <path d="M0,0 L4,2 L0,4 z" fill="var(--c-accent)" />
          </marker>
        </defs>
        {layout.edges.map(edge => {
          const parent = positions.get(edge.parent)
          const child = positions.get(edge.child)
          if (!parent || !child) return null
          return <path key={`${edge.parent}-${edge.child}`}
            d={`M ${parent.x + 5} ${parent.y} C ${parent.x + 12} ${parent.y}, ${child.x - 12} ${child.y}, ${child.x - 5} ${child.y}`}
            markerEnd="url(#dataset-arrow)" />
        })}
      </svg>
      {layout.nodes.map(({ dataset, x, y }) => {
        const active = loadingSource === dataset.source
        const isCloud = dataset.name === 'Cloud Dataset'
        const parents = dataset.parent_sources ?? (dataset.parent_source ? [dataset.parent_source] : [])
        return (
          <article key={dataset.source} className={`dataset-graph__node${isCloud ? ' dataset-graph__node--cloud' : ''}`}
                   style={{ left: `${x}%`, top: `${y}%` }}>
            {dataset.deletable && dataset.id != null && onDelete && (
              <button className="dataset-graph__delete" title="Delete dataset"
                      onClick={() => { if (window.confirm(`Delete dataset "${dataset.name}"? This cannot be undone.`)) onDelete(dataset.id!) }}>
                <Trash2 size={11} />
              </button>
            )}
            <button className="dataset-graph__open" onClick={() => onOpen(dataset)} disabled={loadingSource !== null}>
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
          </article>
        )
      })}
    </div>
  )
}
