import { createContext, useContext } from 'react'

/**
 * Why a demo deployment turns a control off ("Disabled for the Demo"), or `undefined` everywhere else.
 * A control reads it with `useDemoReason()` and, when it is set, disables itself and uses it as its tooltip.
 */
export const DemoContext = createContext<string | undefined>(undefined)

export const useDemoReason = (): string | undefined => useContext(DemoContext)
