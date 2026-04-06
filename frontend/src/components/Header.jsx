import { useDebug } from '../context/DebugContext'

export default function Header() {
  const { enabled, toggle } = useDebug()

  return (
    <header className="bg-cream">
      <div className="max-w-3xl mx-auto flex items-baseline justify-between px-6 pt-10 pb-6">
        <div className="flex items-baseline gap-3">
          <h1 className="font-serif text-2xl font-medium tracking-tight text-ink">
            Catena
          </h1>
          <span className="font-serif italic text-xs text-slate-400 hidden sm:inline">
            a theological study tool
          </span>
        </div>
        <div className="flex items-baseline gap-5">
          <button
            type="button"
            onClick={toggle}
            title="Toggle debug mode (Ctrl+Shift+D)"
            className={`font-mono text-[0.6rem] uppercase tracking-[0.15em] transition-colors ${
              enabled
                ? 'text-accent hover:text-accent-hover'
                : 'text-slate-300 hover:text-slate-500'
            }`}
          >
            {enabled ? 'debug ●' : 'debug ○'}
          </button>
          <a
            href="https://github.com/chriscruz06/ragbench"
            target="_blank"
            rel="noopener noreferrer"
            className="font-serif italic text-xs text-slate-400 hover:text-accent"
          >
            source ↗
          </a>
        </div>
      </div>
      <div className="max-w-3xl mx-auto px-6">
        <hr className="border-rule" />
      </div>
    </header>
  )
}
