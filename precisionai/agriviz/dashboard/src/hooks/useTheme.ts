import { useCallback } from 'react'

const STORAGE_KEY = 'pai-theme'   // shared key across every Precision AI surface

/** Toggles the persisted theme, matching the reference implementation
 * (https://embeddings.precision.ai/ui/tokens.css) exactly: no React state is
 * tracked here — the initial theme is applied before paint by the inline
 * script in index.html, and tokens.css's [data-theme]/prefers-color-scheme CSS
 * (plus the .pai-theme-icon--sun/--moon swap) does all the rendering work. */
export function useTheme() {
  const toggle = useCallback(() => {
    const root = document.documentElement
    const current = root.getAttribute('data-theme')
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
    const next = current === 'dark' ? 'light' : 'dark'
    root.setAttribute('data-theme', next)
    try { localStorage.setItem(STORAGE_KEY, next) } catch { /* ignore */ }
  }, [])

  return { toggle }
}
