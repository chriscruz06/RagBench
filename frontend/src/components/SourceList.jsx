import { Fragment } from 'react'

/**
 * Renders source text with optional highlights. The highlights prop is
 * an array of substrings to mark within the text. Matches are case
 * insensitive. Marked portions get a thin burgundy underline — like
 * someone lightly pencilled under the phrase in a book margin.
 */
function renderHighlighted(text, highlights) {
  if (!highlights?.length) return text

  // Escape regex metacharacters in each highlight
  const escaped = highlights.map((h) =>
    h.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  )
  const pattern = new RegExp(`(${escaped.join('|')})`, 'gi')
  const parts = text.split(pattern)

  return parts.map((part, i) => {
    const isMatch = highlights.some(
      (h) => h.toLowerCase() === part.toLowerCase()
    )
    return isMatch ? (
      <mark
        key={i}
        className="bg-transparent border-b border-accent/40 text-ink"
      >
        {part}
      </mark>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    )
  })
}

/**
 * Extract a stable DOM id from a source's section locator so that
 * in-text citations (like "CCC §1213") can scroll to the matching
 * source. Returns null if the section has no parseable paragraph
 * number — those sources just won't be link targets.
 */
export function getSourceId(section) {
  const match = section?.match(/§\s*(\d+)/)
  return match ? `source-${match[1]}` : null
}

/**
 * SourceList — a footnote-style list of retrieved sources.
 *
 * Each source displays like a scholarly reference: a small numeric
 * marker, italicized title (the work), section locator, and excerpt.
 * Designed to read like the references at the bottom of a page in a
 * theology commentary.
 */
export default function SourceList({ sources }) {
  if (!sources || sources.length === 0) return null

  return (
    <ol className="space-y-9">
      {sources.map((src, i) => {
        const id = getSourceId(src.section)
        return (
          <li
            key={i}
            id={id || undefined}
            className="grid grid-cols-[1.5rem_1fr] gap-x-3 px-2 -mx-2 py-1 -my-1"
          >
            {/* Reference marker — slim and quiet */}
            <span className="font-serif text-sm text-accent/60 tabular-nums pt-[5px]">
              {i + 1}.
            </span>

            <div>
              {/* Title + section locator */}
              <div className="mb-2">
                <span className="font-serif italic text-[0.95rem] text-ink">
                  {src.title}
                </span>
                {src.section && (
                  <>
                    <span className="text-rule mx-2">·</span>
                    <span className="font-serif text-[0.85rem] text-slate-500">
                      {src.section}
                    </span>
                  </>
                )}
              </div>

              {/* Excerpt */}
              <p className="font-serif text-[0.98rem] leading-[1.75] text-slate-600">
                {renderHighlighted(src.text, src.highlights)}
              </p>
            </div>
          </li>
        )
      })}
    </ol>
  )
}
