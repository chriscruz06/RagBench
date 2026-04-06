import { createContext, useContext, useEffect, useState } from 'react'

/**
 * DebugContext — global, persistent debug-mode state.
 *
 * Persisted to localStorage so a page reload preserves the toggle.
 * Keyboard shortcut: Ctrl/Cmd + Shift + D toggles it from anywhere.
 *
 * Usage:
 *   const { enabled, toggle } = useDebug()
 */

const STORAGE_KEY = 'ragbench:debug'

const DebugContext = createContext({
  enabled: false,
  toggle: () => {},
})

export function DebugProvider({ children }) {
  const [enabled, setEnabled] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem(STORAGE_KEY) === 'true'
  })

  // Persist changes
  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, String(enabled))
  }, [enabled])

  // Keyboard shortcut: Ctrl/Cmd + Shift + D
  useEffect(() => {
    function handleKeyDown(e) {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'd') {
        e.preventDefault()
        setEnabled((v) => !v)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const toggle = () => setEnabled((v) => !v)

  return (
    <DebugContext.Provider value={{ enabled, toggle }}>
      {children}
    </DebugContext.Provider>
  )
}

export function useDebug() {
  return useContext(DebugContext)
}
