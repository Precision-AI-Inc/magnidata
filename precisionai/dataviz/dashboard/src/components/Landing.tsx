import { Search, UploadCloud, HelpCircle, ArrowRight } from 'lucide-react'
import './Landing.css'

interface Props {
  onOpenMagniData: () => void
  onOpenByod: () => void
  onOpenHelp: () => void
}

const SECONDARY = [
  {
    key: 'byod',
    icon: UploadCloud,
    title: 'BYOD',
    desc: 'Bring Your Own Data — load a demo dataset or build a new one from your images.',
  },
  {
    key: 'help',
    icon: HelpCircle,
    title: 'Help',
    desc: "How to bring your own data — what files are needed and how they're represented.",
  },
] as const

export function Landing({ onOpenMagniData, onOpenByod, onOpenHelp }: Props) {
  const handlers: Record<string, () => void> = { byod: onOpenByod, help: onOpenHelp }

  return (
    <div className="landing-root">
      {/* Hero — MagniData is the main event on this screen */}
      <button className="landing-hero" onClick={onOpenMagniData}>
        <svg className="landing-hero-motif" viewBox="0 0 400 400" aria-hidden="true">
          <circle cx="260" cy="140" r="46" fill="none" stroke="var(--pai-green)" strokeOpacity="0.35" strokeWidth="1.5" />
          <circle cx="260" cy="140" r="80" fill="none" stroke="var(--pai-fresh)" strokeOpacity="0.22" strokeWidth="1.5" />
          <circle cx="260" cy="140" r="118" fill="none" stroke="var(--pai-green)" strokeOpacity="0.12" strokeWidth="1.5" />
          <circle cx="260" cy="140" r="4" fill="var(--pai-fresh)" />
        </svg>
        <div className="landing-hero-icon"><Search size={26} /></div>
        <div className="landing-hero-body">
          <div className="landing-hero-eyebrow">Main workspace</div>
          <h1>MagniData</h1>
          <p className="landing-hero-tagline">Magnify your data to find what others can't.</p>
          <p className="landing-hero-desc">
            Explore embeddings, cluster and filter your dataset, and inspect every image up
            close — the full visual data-exploration workspace.
          </p>
        </div>
        <span className="landing-hero-cta">Open MagniData <ArrowRight size={14} /></span>
      </button>

      <div className="landing-more-label">More</div>
      <div className="landing-cards">
        {SECONDARY.map(e => {
          const Icon = e.icon
          return (
            <button key={e.key} className="landing-card" onClick={handlers[e.key]}>
              <Icon size={17} className="landing-card-icon" />
              <div className="landing-card-title">{e.title}</div>
              <div className="landing-card-desc">{e.desc}</div>
              <span className="landing-card-link">Open <ArrowRight size={11} /></span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
