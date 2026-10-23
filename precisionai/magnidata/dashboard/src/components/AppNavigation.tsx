import {
  Activity,
  BookOpen,
  BarChart2,
  BarChart3,
  FlaskConical,
  LayoutDashboard,
  ScanSearch,
  Table2,
} from 'lucide-react'
import type { TabId } from '../types'

export type AppSection = 'workspace' | 'explorer' | 'experiments' | 'schema' | 'help'

interface Props {
  active: AppSection
  onWorkspace: () => void
  onExplorer: () => void
  onExperiments: () => void
  onSchema: () => void
  onHelp: () => void
  hasDataset: boolean
  activeTab: TabId
  onDatasetView: (tab: TabId) => void
}

const WORKSPACE_ITEMS = [
  ['workspace', 'Workspace', LayoutDashboard, 'onWorkspace'],
  ['explorer', 'Visual Explorer', ScanSearch, 'onExplorer'],
  ['experiments', 'Experiments', FlaskConical, 'onExperiments'],
] as const

const DATASET_VIEWS: { id: TabId; label: string; icon: typeof ScanSearch }[] = [
  { id: 'visual-explorer', label: 'Explorer', icon: ScanSearch },
  { id: 'charts', label: 'Charts', icon: BarChart2 },
  { id: 'schema', label: 'Schema', icon: BarChart3 },
  { id: 'table', label: 'Table', icon: Table2 },
  { id: 'overview', label: 'Overview', icon: Activity },
]

export function AppNavigation(props: Props) {
  const actions = {
    onWorkspace: props.onWorkspace,
    onExplorer: props.onExplorer,
    onExperiments: props.onExperiments,
    onSchema: props.onSchema,
    onHelp: props.onHelp,
  }

  return (
    <aside className="app-navigation" aria-label="Primary navigation">
      <button className="app-navigation__brand" onClick={props.onWorkspace} aria-label="MagniData workspace">
        <span className="app-navigation__mark"><img src="/logo.png" alt="" /></span>
        <span className="app-navigation__brand-copy">
          <strong>MagniData</strong>
          <small>Precision AI</small>
        </span>
      </button>

      <nav className="app-navigation__items">
        <span className="app-navigation__label">Workspace</span>
        {WORKSPACE_ITEMS.map(([id, label, Icon, action]) => (
          <div key={id} className="app-navigation__group">
            <button
              className="app-navigation__item"
              data-active={props.active === id}
              aria-current={props.active === id ? 'page' : undefined}
              onClick={actions[action]}
            >
              <Icon size={17} strokeWidth={1.8} />
              <span>{label}</span>
            </button>
            {id === 'explorer' && props.hasDataset && (
              <div className="app-navigation__children">
                {DATASET_VIEWS.map(({ id: viewId, label: viewLabel, icon: ViewIcon }) => (
                  <button key={viewId} data-active={props.activeTab === viewId}
                          onClick={() => props.onDatasetView(viewId)}>
                    <ViewIcon size={13} /><span>{viewLabel}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        <span className="app-navigation__label app-navigation__label--support">Support</span>
        {!props.hasDataset && (
          <button
            className="app-navigation__item"
            data-active={props.active === 'schema'}
            aria-current={props.active === 'schema' ? 'page' : undefined}
            onClick={props.onSchema}
          >
            <BarChart3 size={18} strokeWidth={1.8} />
            <span>Data Schema</span>
          </button>
        )}
        <button className="app-navigation__item" data-active={props.active === 'help'}
                aria-current={props.active === 'help' ? 'page' : undefined} onClick={props.onHelp}>
          <BookOpen size={18} strokeWidth={1.8} /><span>User Guide</span>
        </button>
      </nav>

      <div className="app-navigation__footer">
        <span className="app-navigation__footer-icon"><Activity size={15} /></span>
        <span><strong>Local workspace</strong><small><i className="app-navigation__status-dot" /> Services available</small></span>
      </div>
    </aside>
  )
}
