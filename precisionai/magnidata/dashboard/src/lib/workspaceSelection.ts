export const SELECTED_KEY = 'magnidata.workspace.selected'

/**
 * Next selection after clicking a dataset. A plain click selects only that dataset (or clears it when it
 * is the sole selection); an additive click (Shift/Ctrl/Cmd) toggles it within a multi-selection.
 */
export function nextSelection(selection: string[], source: string, additive: boolean): string[] {
  if (additive) return selection.includes(source) ? selection.filter(item => item !== source) : [...selection, source]
  return selection.length === 1 && selection[0] === source ? [] : [source]
}

export function canMerge(selection: string[]): boolean {
  return new Set(selection).size >= 2
}

export function rememberWorkspaceSelection(source: string | null): void {
  try {
    if (source) localStorage.setItem(SELECTED_KEY, source)
    else localStorage.removeItem(SELECTED_KEY)
  } catch { /* storage unavailable: selection just won't survive a reload */ }
}

export function readWorkspaceSelection(): string | null {
  try { return localStorage.getItem(SELECTED_KEY) } catch { return null }
}
