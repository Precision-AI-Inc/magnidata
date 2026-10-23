export interface LineageInput {
  datasetName: string | null | undefined
  parentRunId: string | null | undefined
  parentRunName?: string | null
}

/** "Trained on <dataset>, from run <parent>" — or null when the run has no parent. */
export function lineageLabel({ datasetName, parentRunId, parentRunName }: LineageInput): string | null {
  if (!parentRunId) return null
  const parent = parentRunName || parentRunId.slice(0, 8)
  return `Trained on ${datasetName || 'the dataset'}, from run ${parent}`
}

/** Short text for the optional "Parent run" table column. */
export function parentColumnText(parentRunId: string | null | undefined, names: Record<string, string>): string {
  if (!parentRunId) return '—'
  return names[parentRunId] || parentRunId.slice(0, 8)
}
