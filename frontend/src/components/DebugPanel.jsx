/**
 * DebugPanel — developer-only evaluation metrics.
 *
 * Visually distinct from the main reading UI: monospace font, dense
 * spacing, no serifs. The aesthetic break is intentional — it
 * signals "this is engineering surface, not part of the study tool".
 *
 * Hidden by default; revealed only when DebugContext.enabled is true.
 */

const METRIC_DEFS = [
  {
    key: 'relevance',
    label: 'Relevance',
    description: 'Mean similarity score across retrieved chunks (0–1)',
  },
  {
    key: 'faithfulness',
    label: 'Faithfulness',
    description: 'Token overlap between answer and retrieved context',
  },
  {
    key: 'context_utilization',
    label: 'Context Utilization',
    description: 'Fraction of retrieved chunks cited in the answer',
  },
]

function formatScore(value) {
  if (value === null || value === undefined) return '—'
  return Number(value).toFixed(3)
}

export default function DebugPanel({ metrics, chunksUsed }) {
  if (!metrics) return null

  return (
    <section className="mt-16 pt-5 border-t border-dashed border-slate-300">
      <div className="flex items-baseline justify-between mb-4">
        <h3 className="font-mono text-[0.65rem] uppercase tracking-[0.15em] text-slate-400">
          debug · evaluation metrics
        </h3>
        <span className="font-mono text-[0.65rem] text-slate-400 tabular-nums">
          {chunksUsed} chunks
        </span>
      </div>

      <table className="w-full font-mono text-xs border border-slate-200 bg-white/40">
        <thead>
          <tr className="border-b border-slate-200 text-slate-400">
            <th className="text-left px-3 py-2 font-normal w-1/2">metric</th>
            <th className="text-right px-3 py-2 font-normal">score</th>
          </tr>
        </thead>
        <tbody>
          {METRIC_DEFS.map((m, i) => (
            <tr
              key={m.key}
              className={
                i < METRIC_DEFS.length - 1 ? 'border-b border-slate-200' : ''
              }
            >
              <td className="px-3 py-2 align-top">
                <div className="text-slate-700">{m.label}</div>
                <div className="text-slate-400 text-[0.65rem] mt-0.5 leading-snug">
                  {m.description}
                </div>
              </td>
              <td className="px-3 py-2 text-right text-slate-800 tabular-nums align-top">
                {formatScore(metrics[m.key])}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="mt-3 font-mono text-[0.6rem] text-slate-400">
        intrinsic metrics — no ground truth required. press ⌘/ctrl+shift+d to
        toggle.
      </p>
    </section>
  )
}
