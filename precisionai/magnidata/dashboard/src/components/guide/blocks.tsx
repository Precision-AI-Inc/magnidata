import { useState } from 'react'
import type { ReactNode } from 'react'
import { AlertTriangle, Check, Copy, Info, Lightbulb } from 'lucide-react'
import type { Callout } from '../../lib/guide'
import shots from './shots.json'
import { COMMANDS } from './commands'
import type { CommandKey } from './commands'

export type ShotName = keyof typeof shots

/** A copyable shell command from the shared command list. */
export function Cmd({ id, note }: { id: CommandKey; note?: string }) {
  const [copied, setCopied] = useState(false)
  const text = COMMANDS[id]
  const copy = () => {
    const done = () => { setCopied(true); window.setTimeout(() => setCopied(false), 1500) }
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => undefined)
  }
  return (
    <div className="guide-cmd">
      {note && <span className="guide-cmd__note">{note}</span>}
      <pre><code>{text}</code></pre>
      <button onClick={copy} aria-label={`Copy command: ${text.split('\n')[0]}`} title="Copy">
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
    </div>
  )
}

/** Terminal output the reader should expect (not copyable, not a command). */
export function Output({ children }: { children: string }) {
  return <pre className="guide-output"><code>{children}</code></pre>
}

/** A real screenshot from the UI with optional numbered callouts drawn in CSS. */
export function Shot({ name, callouts = [] }: { name: ShotName; callouts?: Callout[] }) {
  const meta = shots[name]
  return (
    <figure className="guide-shot">
      <div className="guide-shot__frame">
        <img src={`/guide/${name}.png`} alt={meta.alt} loading="lazy" />
        {callouts.map((callout, index) => (
          <span key={callout.text} className="guide-shot__badge" style={{ left: `${callout.x}%`, top: `${callout.y}%` }}>
            {index + 1}
          </span>
        ))}
      </div>
      <figcaption>
        {meta.caption}
        {callouts.length > 0 && (
          <ol>{callouts.map(callout => <li key={callout.text}>{callout.text}</li>)}</ol>
        )}
      </figcaption>
    </figure>
  )
}

export function Steps({ children }: { children: ReactNode }) {
  return <ol className="guide-steps">{children}</ol>
}

/** What the reader should see once a step is done, so they can check themselves. */
export function Expect({ children }: { children: ReactNode }) {
  return (
    <aside className="guide-expect">
      <strong><Check size={14} /> What you should see</strong>
      <div>{children}</div>
    </aside>
  )
}

export function Note({ tone = 'info', title, children }: { tone?: 'info' | 'tip' | 'warn'; title?: string; children: ReactNode }) {
  const Icon = tone === 'warn' ? AlertTriangle : tone === 'tip' ? Lightbulb : Info
  return (
    <aside className={`guide-note guide-note--${tone}`}>
      <Icon size={16} />
      <div>{title && <strong>{title}</strong>}{children}</div>
    </aside>
  )
}

export function Goal({ children }: { children: ReactNode }) {
  return <p className="guide-goal">{children}</p>
}

export function Term({ children }: { children: ReactNode }) {
  return <em className="guide-term">{children}</em>
}
