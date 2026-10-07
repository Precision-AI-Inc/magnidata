import {
  BarChart3,
  CircleHelp,
  Database,
  FlaskConical,
  LayoutDashboard,
  ScanSearch,
} from 'lucide-react'

export type AppSection = 'workspace' | 'explorer' | 'datasets' | 'experiments' | 'schema' | 'help'

interface Props {
  active: AppSection
  onWorkspace: () => void
  onExplorer: () => void
  onDatasets: () => void
  onExperiments: () => void
  onSchema: () => void
  onHelp: () => void
}

const ITEMS = [
  ['workspace', 'Workspace', LayoutDashboard, 'onWorkspace'],
  ['explorer', 'Visual Explorer', ScanSearch, 'onExplorer'],
  ['datasets', 'Datasets', Database, 'onDatasets'],
  ['experiments', 'Experiments', FlaskConical, 'onExperiments'],
  ['schema', 'Data Schema', BarChart3, 'onSchema'],
  ['help', 'Help & Setup', CircleHelp, 'onHelp'],
] as const

export function AppNavigation(props: Props) {
  const actions = {
    onWorkspace: props.onWorkspace,
    onExplorer: props.onExplorer,
    onDatasets: props.onDatasets,
    onExperiments: props.onExperiments,
    onSchema: props.onSchema,
    onHelp: props.onHelp,
  }

  return (
    <aside className="app-navigation" aria-label="Primary navigation">
      <button className="app-navigation__brand" onClick={props.onWorkspace} aria-label="MagniData workspace">
        <span className="app-navigation__mark"><img src="/logo.png" alt="" /></span>
        <span>MagniData</span>
      </button>

      <nav className="app-navigation__items">
        {ITEMS.map(([id, label, Icon, action]) => (
          <button
            key={id}
            className="app-navigation__item"
            data-active={props.active === id}
            aria-current={props.active === id ? 'page' : undefined}
            onClick={actions[action]}
          >
            <Icon size={17} strokeWidth={1.8} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <div className="app-navigation__footer">
        <span className="app-navigation__status-dot" />
        Local workspace
      </div>
    </aside>
  )
}
