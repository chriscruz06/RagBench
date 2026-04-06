import { useState, useEffect } from 'react'
import QuestionInput from '../components/QuestionInput'
import AnswerView from '../components/AnswerView'
import DebugPanel from '../components/DebugPanel'
import { useDebug } from '../context/DebugContext'

const EXAMPLE_QUESTIONS = [
  'What is the Trinity?',
  'What is the Eucharist?',
  'What is Baptism?',
  'What is Original Sin?',
]

const LOADING_MESSAGES = [
  'Searching the sources…',
  'Reading the passages…',
  'Weighing the witnesses…',
  'Composing an answer…',
]

export default function Study() {
  const [questionValue, setQuestionValue] = useState('')
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [loadingMsgIdx, setLoadingMsgIdx] = useState(0)
  const { enabled: debugEnabled } = useDebug()

  // Cycle through loading messages every ~3.5s while a query is in flight.
  // Resets to the first message whenever loading toggles off.
  useEffect(() => {
    if (!loading) {
      setLoadingMsgIdx(0)
      return
    }
    const interval = setInterval(() => {
      setLoadingMsgIdx((i) => (i + 1) % LOADING_MESSAGES.length)
    }, 3500)
    return () => clearInterval(interval)
  }, [loading])

  async function handleAsk(question) {
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 5 }),
      })

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`)
      }

      const data = await res.json()
      setResponse(data)
    } catch (err) {
      setError(
        err.message.includes('Failed to fetch')
          ? 'The sources cannot be reached. The backend may not be running.'
          : `An error occurred: ${err.message}`
      )
    } finally {
      setLoading(false)
    }
  }

  function handleExampleClick(text) {
    setQuestionValue(text)
    handleAsk(text)
  }

  return (
    <div>
      <QuestionInput
        value={questionValue}
        onChange={setQuestionValue}
        onAsk={handleAsk}
        loading={loading}
      />

      {/* Error state — quiet and reserved */}
      {error && (
        <div className="mt-16 pt-6 border-t border-rule">
          <p className="font-serif italic text-sm text-accent mb-2">
            Unable to answer.
          </p>
          <p className="font-serif text-[0.95rem] text-slate-600 leading-relaxed">
            {error}
          </p>
        </div>
      )}

      {/* Loading state — cycling messages, no spinner */}
      {loading && (
        <div className="mt-24 text-center">
          <p
            key={loadingMsgIdx}
            className="font-serif italic text-slate-400 text-base"
          >
            {LOADING_MESSAGES[loadingMsgIdx]}
          </p>
        </div>
      )}

      {/* Successful response */}
      {response && !loading && <AnswerView response={response} />}

      {/* Developer debug panel — only when debug mode is on */}
      {response && !loading && debugEnabled && (
        <DebugPanel
          metrics={response.debug_metrics}
          chunksUsed={response.chunks_used}
        />
      )}

      {/* Empty state — invitation + clickable starter questions */}
      {!response && !loading && !error && (
        <div className="mt-28">
          <p className="font-serif italic text-slate-300 text-base text-center mb-10">
            Or begin with one of these.
          </p>
          <ul className="flex flex-col items-center gap-3">
            {EXAMPLE_QUESTIONS.map((q) => (
              <li key={q}>
                <button
                  type="button"
                  onClick={() => handleExampleClick(q)}
                  className="font-serif italic text-base text-slate-500 hover:text-accent transition-colors"
                >
                  {q}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
