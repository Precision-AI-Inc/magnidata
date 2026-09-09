import { useState, type CSSProperties, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { COL_DESCS, GROUP_COLORS } from '../data/colDescs'

interface Props {
  col: string
  children: ReactNode
  style?: CSSProperties
  className?: string
}

export function ColTooltip({ col, children, style, className }: Props) {
  const desc = COL_DESCS[col]
  const [rect, setRect] = useState<DOMRect | null>(null)

  if (!desc) return <span style={style} className={className}>{children}</span>

  const groupColor = GROUP_COLORS[desc.group] ?? '#64748b'

  return (
    <>
      <span
        className={className}
        style={{
          ...style,
          cursor: 'help',
          borderBottom: `1px dashed ${groupColor}55`,
          textDecoration: 'none',
        }}
        onMouseEnter={e => setRect(e.currentTarget.getBoundingClientRect())}
        onMouseMove={e => setRect(e.currentTarget.getBoundingClientRect())}
        onMouseLeave={() => setRect(null)}
      >
        {children}
      </span>

      {rect && createPortal(
        <div style={{
          position: 'fixed',
          left: Math.min(rect.left + rect.width / 2, window.innerWidth - 320),
          top: rect.top - 10,
          transform: 'translateY(-100%)',
          background: '#0d2333',
          color: '#e2e8f0',
          borderRadius: 10,
          padding: '10px 14px',
          width: 300,
          zIndex: 99999,
          fontSize: 12,
          lineHeight: 1.55,
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          pointerEvents: 'none',
          border: `1px solid ${groupColor}44`,
        }}>
          {/* Group badge */}
          <div style={{
            display: 'inline-block',
            fontSize: 9, fontWeight: 700,
            textTransform: 'uppercase', letterSpacing: '.6px',
            padding: '2px 7px', borderRadius: 8, marginBottom: 6,
            background: groupColor + '33',
            color: groupColor === '#013755' ? '#6DF2A3' : groupColor,
            border: `1px solid ${groupColor}44`,
          }}>
            {desc.group}
          </div>

          {/* Column name */}
          <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 5,
                        fontFamily: 'JetBrains Mono, monospace', color: '#fff' }}>
            {col}
          </div>

          {/* Meaning */}
          <div style={{ color: '#cbd5e1', marginBottom: 7 }}>
            {desc.plain_meaning}
          </div>

          {/* Scale */}
          <div style={{
            fontSize: 10, color: '#94a3b8', fontStyle: 'italic',
            paddingTop: 5, borderTop: '1px solid rgba(255,255,255,0.08)',
          }}>
            {desc.scale}
          </div>
        </div>,
        document.body,
      )}
    </>
  )
}
