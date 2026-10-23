/** The services and the path data takes through them. Drawn inline so it follows the theme and needs no asset. */
export function Architecture() {
  const box = (x: number, y: number, w: number, title: string, lines: string[], tone = 'var(--c-accent)') => (
    <g key={title}>
      <rect x={x} y={y} width={w} height={74} rx={12} fill="var(--c-surface)" stroke={tone} strokeWidth={1.6} />
      <text x={x + w / 2} y={y + 26} textAnchor="middle" className="guide-svg-title">{title}</text>
      {lines.map((line, index) => (
        <text key={line} x={x + w / 2} y={y + 46 + index * 14} textAnchor="middle" className="guide-svg-text">{line}</text>
      ))}
    </g>
  )
  const arrow = (x1: number, y1: number, x2: number, y2: number, label?: string) => (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--c-t3)" strokeWidth={1.4} markerEnd="url(#guide-arrow)" />
      {label && (x1 === x2
        ? <text x={x1 + 8} y={(y1 + y2) / 2 + 3} className="guide-svg-label">{label}</text>
        : <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle" className="guide-svg-label">{label}</text>)}
    </g>
  )
  return (
    <svg viewBox="0 0 760 330" role="img" className="guide-diagram"
         aria-label="Your browser talks to the web portal, which proxies the API; the API reads and writes the data folders and MLflow; the trainer worker picks up scheduled runs from MLflow.">
      <defs>
        <marker id="guide-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--c-t3)" />
        </marker>
      </defs>
      {box(20, 20, 150, 'Your browser', ['the dashboard UI'], 'var(--c-t3)')}
      {box(230, 20, 160, 'web  :5175', ['nginx + React portal', 'serves this guide'])}
      {box(450, 20, 160, 'api  :5051', ['Flask: datasets, images,', 'projections, run browsing'])}
      {box(450, 130, 160, 'mlflow  :5052', ['runs, metrics,', 'predictions, checkpoints'], '#d68013')}
      {box(450, 240, 160, 'trainer', ['picks up scheduled runs', 'CPU, or GPU if enabled'], '#d68013')}
      {box(660, 20, 90, 'data', ['CSVs · tiles', 'image_sets/'], '#13a77f')}
      {arrow(170, 57, 230, 57)}
      {arrow(390, 57, 450, 57, '/api')}
      {arrow(610, 57, 660, 57)}
      {arrow(530, 94, 530, 130, 'runs')}
      {arrow(530, 204, 530, 240, 'schedule')}
      <text x={20} y={150} className="guide-svg-label">Data flow</text>
      {['images', 'features (+ embeddings)', 'dataset CSV', 'lineage graph', 'explore', 'train', 'per-image loss', 'curate'].map((step, index) => (
        <g key={step}>
          <rect x={20 + (index % 2) * 196} y={162 + Math.floor(index / 2) * 40} width={180} height={28} rx={14} fill="var(--c-accent-light)" stroke="var(--c-accent)" />
          <text x={110 + (index % 2) * 196} y={181 + Math.floor(index / 2) * 40} textAnchor="middle" className="guide-svg-text">{index + 1}. {step}</text>
        </g>
      ))}
    </svg>
  )
}
