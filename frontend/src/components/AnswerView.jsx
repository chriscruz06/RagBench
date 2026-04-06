import { useState, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import SourceList from './SourceList'

function SectionLabel({ children }) {
  return (
    <h3 className="font-serif italic text-base text-slate-500 mb-6">
      {children}
    </h3>
  )
}

/**
 * Preprocess the answer text to convert citation references like
 * "CCC §1213" or bare "§985" into markdown links pointing at the
 * matching source's DOM id. Only links references whose paragraph
 * number actually appears in the retrieved sources — otherwise the
 * text is left untouched.
 */
function linkifyCitations(answer, sources) {
  const sourceParaNums = new Set()
  sources.forEach((s) => {
    const m = s.section?.match(/§\s*(\d+)/)
    if (m) sourceParaNums.add(m[1])
  })

  if (sourceParaNums.size === 0) return answer

  // Match optional "CCC " prefix + § + optional space + digits
  return answer.replace(/(CCC\s+)?§\s*(\d+)/g, (full, _prefix, num) => {
    if (sourceParaNums.has(num)) {
      return `[${full}](#source-${num})`
    }
    return full
  })
}

/**
 * Click handler for in-text citation links.
 * Scrolls to the matching source and briefly flashes its background.
 */
function handleCitationClick(e, href) {
  e.preventDefault()
  const id = href.slice(1) // strip leading #
  const el = document.getElementById(id)
  if (!el) return

  el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  // Reset any in-flight flash before re-applying
  el.classList.remove('citation-flash')
  // eslint-disable-next-line no-unused-expressions
  el.offsetWidth // force reflow so the animation restarts
  el.classList.add('citation-flash')
  setTimeout(() => el.classList.remove('citation-flash'), 1700)
}

// ── Markdown component overrides ─────────────────────────
// Keep everything in the contemplative palette. No headings —
// the model rarely emits them and we don't want them anyway.
const markdownComponents = {
  // Paragraphs handled via .dropcap CSS — no className needed here
  p: ({ children }) => <p>{children}</p>,
  strong: ({ children }) => (
    <strong className="font-medium text-ink">{children}</strong>
  ),
  em: ({ children }) => <em className="italic">{children}</em>,
  ul: ({ children }) => (
    <ul className="list-disc pl-6 mt-3 space-y-1.5">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal pl-6 mt-3 space-y-1.5">{children}</ol>
  ),
  li: ({ children }) => <li className="pl-1">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="pl-4 my-4 border-l border-rule italic text-slate-600">
      {children}
    </blockquote>
  ),
  a: ({ href, children }) => {
    if (href?.startsWith('#source-')) {
      return (
        <a
          href={href}
          onClick={(e) => handleCitationClick(e, href)}
          className="text-accent hover:text-accent-hover no-underline border-b border-accent/40 hover:border-accent transition-colors"
        >
          {children}
        </a>
      )
    }
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-accent underline"
      >
        {children}
      </a>
    )
  },
}

export default function AnswerView({ response }) {
  const [showGrounding, setShowGrounding] = useState(false)

  // Memoize the linkified answer so we don't reprocess on every render
  const processedAnswer = useMemo(
    () => linkifyCitations(response.answer, response.sources || []),
    [response.answer, response.sources]
  )

  return (
    <div className="mt-20 space-y-20">
      {/* ── Primary: the answer ─────────────────── */}
      <section>
        <SectionLabel>Answer.</SectionLabel>
        <div className="dropcap font-serif text-[1.2rem] leading-[1.85] text-ink">
          <ReactMarkdown components={markdownComponents}>
            {processedAnswer}
          </ReactMarkdown>
        </div>
      </section>

      {/* ── Sources (visually separated, like footnotes) ── */}
      {response.sources && response.sources.length > 0 && (
        <section className="pt-12 border-t border-rule">
          <SectionLabel>Sources.</SectionLabel>
          <SourceList sources={response.sources} />
        </section>
      )}

      {/* ── Optional grounding / reasoning ──────── */}
      {response.grounding && (
        <section>
          <button
            onClick={() => setShowGrounding((s) => !s)}
            className="font-serif italic text-sm text-slate-400 hover:text-accent transition-colors"
          >
            {showGrounding ? '— Hide grounding' : '+ Show grounding'}
          </button>
          {showGrounding && (
            <p className="mt-5 font-serif italic text-[1rem] text-slate-600 leading-[1.8]">
              {response.grounding}
            </p>
          )}
        </section>
      )}
    </div>
  )
}
