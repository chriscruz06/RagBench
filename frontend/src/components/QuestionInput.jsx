/**
 * QuestionInput — controlled by the parent (Study).
 *
 * The parent owns the textarea value so example questions can
 * pre-fill it and so the input can be reset programmatically.
 */
export default function QuestionInput({ value, onChange, onAsk, loading }) {
  function submit() {
    const trimmed = value.trim()
    if (trimmed && !loading) {
      onAsk(trimmed)
    }
  }

  // Submit on Ctrl/Cmd+Enter for multiline comfort
  function handleKeyDown(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="w-full">
      <label
        htmlFor="question"
        className="block font-serif italic text-sm text-slate-500 mb-3"
      >
        Pose a question.
      </label>
      <textarea
        id="question"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="What is Baptism?"
        rows={2}
        disabled={loading}
        className="w-full px-1 py-2 font-serif text-xl text-ink bg-transparent border-0 border-b border-rule focus:outline-none focus:border-accent placeholder:text-slate-300 placeholder:italic resize-none disabled:opacity-60"
      />
      <div className="mt-4 flex justify-end">
        <button
          type="button"
          onClick={submit}
          disabled={loading || !value.trim()}
          className="font-serif italic text-base text-accent hover:text-accent-hover disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Consulting…' : 'Ask →'}
        </button>
      </div>
    </div>
  )
}
