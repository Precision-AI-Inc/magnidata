import { describe, expect, it } from 'vitest'
import { layoutDatasetGraph } from './datasetGraph'

describe('layoutDatasetGraph', () => {
  it('places cloud, mini datasets, and a two-parent derived dataset in successive columns', () => {
    const layout = layoutDatasetGraph([
      { name: 'Cloud', description: '', source: 'cloud' },
      { name: 'Mini 1', description: '', source: 'mini1', parent_sources: ['cloud'] },
      { name: 'Mini 2', description: '', source: 'mini2', parent_sources: ['cloud'] },
      { name: 'Curated', description: '', source: 'curated', parent_sources: ['mini1', 'cloud'] },
    ])
    expect(Object.fromEntries(layout.nodes.map(node => [node.dataset.source, node.depth]))).toEqual({
      cloud: 0, mini1: 1, mini2: 1, curated: 2,
    })
    expect(layout.edges).toContainEqual({ parent: 'mini1', child: 'curated' })
    expect(layout.edges).toContainEqual({ parent: 'cloud', child: 'curated' })
    const positions = Object.fromEntries(layout.nodes.map(node => [node.dataset.source, node]))
    expect(positions.cloud.x).toBeLessThan(positions.mini1.x)
    expect(positions.mini1.x).toBeLessThan(positions.curated.x)
    expect(positions.mini1.y).not.toBe(positions.mini2.y)
  })

  it('breaks accidental lineage cycles', () => {
    const layout = layoutDatasetGraph([
      { name: 'A', description: '', source: 'a', parent_sources: ['b'] },
      { name: 'B', description: '', source: 'b', parent_sources: ['a'] },
    ])
    expect(layout.nodes).toHaveLength(2)
  })

  it('expands its bounds for deep and wide lineage graphs', () => {
    const narrow = layoutDatasetGraph([
      { name: 'Root', description: '', source: 'root' },
      { name: 'Child', description: '', source: 'child', parent_sources: ['root'] },
    ])
    const deep = layoutDatasetGraph([
      { name: 'Root', description: '', source: 'root' },
      { name: 'Child', description: '', source: 'child', parent_sources: ['root'] },
      { name: 'Grandchild', description: '', source: 'grandchild', parent_sources: ['child'] },
    ])
    const wide = layoutDatasetGraph([
      { name: 'Root', description: '', source: 'root' },
      { name: 'A', description: '', source: 'a', parent_sources: ['root'] },
      { name: 'B', description: '', source: 'b', parent_sources: ['root'] },
      { name: 'C', description: '', source: 'c', parent_sources: ['root'] },
    ])

    expect(deep.width).toBeGreaterThan(narrow.width)
    expect(wide.height).toBeGreaterThan(narrow.height)
  })
})
