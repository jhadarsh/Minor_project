import { useEffect, useRef, useState } from 'react'
import { api, fileForm } from './api'
import { toWav } from './audio'
import { useRecording } from './useRecording'
import './App.css'

function Wave({ className = '' }) {
  return (
    <span className={`wave ${className}`} aria-hidden="true">
      {[24, 46, 72, 100, 66, 40, 22].map((size, i) => <i key={i} style={{ height: `${size}%` }} />)}
    </span>
  )
}

function Notice({ children }) {
  return children ? <p className="notice" role="alert">{children}</p> : null
}

function Credential({ label, value }) {
  const [feedback, setFeedback] = useState('')
  async function copy() {
    try {
      await navigator.clipboard.writeText(value)
      setFeedback('Copied')
    } catch {
      setFeedback('Select the key to copy it')
    }
  }
  return (
    <div className="credential">
      <label>{label}<input value={value} readOnly onFocus={(event) => event.target.select()} /></label>
      <button type="button" className="secondary" onClick={copy}>Copy</button>
      <span className="feedback" role="status">{feedback}</span>
    </div>
  )
}

function VoiceUpload({ active, uploadKey }) {
  const [file, setFile] = useState(null)
  const [consent, setConsent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')

  const [previewUrl, setPreviewUrl] = useState('')
  const fileInput = useRef(null)
  const { record, recording, opening } = useRecording({
    active,
    maxSeconds: 20,
    onFile(recorded) {
      fileInput.current.value = ''
      selectFile(recorded)
    },
    onError: setError,
  })
  const capturing = recording || opening

  useEffect(() => {
    const url = file ? URL.createObjectURL(file) : ''
    setPreviewUrl(url)
    return () => { if (url) URL.revokeObjectURL(url) }
  }, [file])

  function selectFile(selected) {
    setFile(selected)
    setStatus('')
    setError('')
    setConsent(false)
  }

  async function upload(event) {
    event.preventDefault()
    if (busy || capturing || !file || !consent || !uploadKey.trim()) return
    setBusy(true)
    setStatus('')
    setError('')
    try {
      const form = fileForm(await toWav(file, 60))
      form.append('consent', String(consent))
      const response = await api('voice', uploadKey, form)
      setStatus(`Reference voice saved · ${response.seconds} seconds`)
    } catch (failure) {
      setError(failure.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card voice-card">
      <h2>Organization voice</h2>
      <p className="muted">Record or upload 10–20 seconds of clear speech from one person. A new recording replaces the organization’s reference voice.</p>
      <form className="stack" onSubmit={upload}>
        <div className="actions">
          <button type="button" className={recording ? 'stop' : 'secondary'} onClick={record} disabled={busy || opening || !uploadKey.trim()}>
            <Wave />{recording ? 'Stop recording' : 'Record voice'}
          </button>
        </div>
        {opening && <p className="muted" role="status">Opening microphone…</p>}
        {recording && <p className="recording" role="status">Recording · Up to 20 seconds</p>}
        <label className="file-picker">
          <span>Choose a reference recording</span>
          <input ref={fileInput} type="file" accept="audio/*,.webm,.m4a" disabled={busy || capturing} onChange={(event) => {
            selectFile(event.target.files[0] ?? null)
          }} />
        </label>
        {previewUrl && !capturing && <audio key={previewUrl} controls src={previewUrl} aria-label="Preview reference voice" />}
        <label className="checkbox-row">
          <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} disabled={busy || capturing} />
          I have permission from the speaker to clone this voice.
        </label>
        <Notice>{error}</Notice>
        {status && <p role="status">{status}</p>}
        <button className="primary" disabled={busy || capturing || !file || !consent || !uploadKey.trim()}>{busy ? 'Saving…' : 'Save reference voice'}</button>
        {!uploadKey.trim() && <p className="muted">Create an organization or enter its upload key above.</p>}
      </form>
    </section>
  )
}

function AdminPage({ active, onTry }) {
  const [organization, setOrganization] = useState(null)
  const [adminKey, setAdminKey] = useState('')
  const [name, setName] = useState('')
  const [uploadKey, setUploadKey] = useState('')
  const [file, setFile] = useState(null)
  const [documents, setDocuments] = useState([])
  const [creating, setCreating] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [createError, setCreateError] = useState('')
  const [uploadError, setUploadError] = useState('')
  const fileInput = useRef(null)

  async function createOrganization(event) {
    event.preventDefault()
    setCreating(true)
    setCreateError('')
    try {
      const result = await api('organizations', adminKey, { name: name.trim() })
      setOrganization(result)
      setUploadKey(result.upload_key)
      setDocuments([])
      setAdminKey('')
    } catch (error) {
      setCreateError(error.message)
    } finally {
      setCreating(false)
    }
  }

  async function uploadDocument(event) {
    event.preventDefault()
    if (!file) return
    setUploading(true)
    setUploadError('')
    try {
      const result = await api('documents', uploadKey, fileForm(file))
      setDocuments((previous) => [...previous.filter((item) => item.id !== result.id), result])
      setFile(null)
      fileInput.current.value = ''
    } catch (error) {
      setUploadError(error.message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <>
      <section className="intro">
        <h1>Admin workspace</h1>
        <p>Upload the documents your assistant will use and get API keys for your organization.</p>
      </section>
      <div className="workspace">
        <section className="card">
          <h2>Organization</h2>
          {organization ? (
            <div className="stack">
              <p className="organization-name">{organization.name}</p>
              <p className="muted">Save both keys now. They’re shown only when you create the organization.</p>
              <Credential key={organization.upload_key} label="Upload API key" value={organization.upload_key} />
              <Credential key={organization.query_key} label="Query API key" value={organization.query_key} />
              <div className="actions">
                <button type="button" className="primary" onClick={() => onTry(organization.query_key)}>Try the assistant</button>
                <button type="button" className="secondary" disabled={uploading} onClick={() => {
                  setOrganization(null)
                  setUploadKey('')
                  setDocuments([])
                  setName('')
                }}>Set up another</button>
              </div>
            </div>
          ) : (
            <form className="stack" onSubmit={createOrganization}>
              <label>Organization name<input required maxLength={150} value={name} onChange={(event) => setName(event.target.value)} placeholder="Enter organization name" /></label>
              <label>Platform admin key<input required type="password" autoComplete="off" value={adminKey} onChange={(event) => setAdminKey(event.target.value)} placeholder="Enter admin key" /></label>
              <Notice>{createError}</Notice>
              <button className="primary" disabled={creating || uploading || !name.trim() || !adminKey.trim()}>{creating ? 'Creating…' : 'Create organization'}</button>
            </form>
          )}
        </section>
        <section className="card">
          <h2>Upload a document</h2>
          <form className="stack" onSubmit={uploadDocument}>
            <label>Upload API key<input required type="password" autoComplete="off" value={uploadKey} disabled={uploading} onChange={(event) => {
              setUploadKey(event.target.value)
              setDocuments([])
            }} placeholder="Enter organization upload key" /></label>
            <label className="file-picker">
              <span>Choose a document</span>
              <small>Searchable PDF, TXT, or Markdown · Up to 10 MiB</small>
              <input ref={fileInput} type="file" accept=".pdf,.txt,.md" disabled={uploading} onChange={(event) => setFile(event.target.files[0] ?? null)} />
            </label>
            <Notice>{uploadError}</Notice>
            <button className="primary" disabled={uploading || creating || !file || !uploadKey.trim()}>{uploading ? 'Indexing…' : 'Upload document'}</button>
          </form>
          {documents.length > 0 && (
            <ul className="documents" aria-label="Indexed documents" aria-live="polite">
              {documents.map((document) => <li key={document.id}><strong>{document.filename}</strong><span>Indexed · {document.chunk_count} excerpts</span></li>)}
            </ul>
          )}
        </section>
      </div>
      <VoiceUpload key={uploadKey} active={active} uploadKey={uploadKey} />
      <section className="integration">
        <div><h2>Connect your platform</h2><p>Use the query key on your server to ask questions, transcribe audio, or retrieve spoken answers.</p></div>
        <ul>
          <li><code>POST /v1/documents</code><span>Upload key</span></li>
          <li><code>POST /v1/voice</code><span>Upload key</span></li>
          <li><code>POST /v1/ask</code><span>Query key</span></li>
          <li><code>POST /v1/transcribe</code><span>Query key</span></li>
          <li><code>GET /v1/answers/&#123;audio_id&#125;/audio</code><span>Query key</span></li>
        </ul>
      </section>
    </>
  )
}

function AnswerAudio({ result, queryKey }) {
  const [audioUrl, setAudioUrl] = useState('')
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (!result.audio_id) return
    let url = ''
    let active = true
    const start = window.setTimeout(() => {
      api(`answers/${result.audio_id}/audio`, queryKey).then((audio) => {
        if (active) {
          url = URL.createObjectURL(audio)
          setAudioUrl(url)
        }
      }).catch((failure) => {
        if (active) setError(failure.message)
      })
    }, 0)
    return () => {
      active = false
      window.clearTimeout(start)
      if (url) URL.revokeObjectURL(url)
    }
  }, [result.audio_id, queryKey, attempt])

  if (!result.audio_id) return result.audio_message && <p className="muted audio-status">{result.audio_message}</p>
  return (
    <div className="audio-status">
      <p className="muted">{result.audio_voice}</p>
      {audioUrl ? <audio controls autoPlay src={audioUrl} aria-label="Spoken answer" /> : error ? (
        <><Notice>{error}</Notice><button className="secondary" type="button" onClick={() => { setError(''); setAttempt(attempt + 1) }}>Retry audio</button></>
      ) : <p role="status">Preparing spoken answer…</p>}
    </div>
  )
}

function QuestionPage({ queryKey, setQueryKey, active }) {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [voiceLanguages, setVoiceLanguages] = useState({})
  const [voiceLanguage, setVoiceLanguage] = useState('')
  const [transcriptLanguage, setTranscriptLanguage] = useState(null)
  const audioInput = useRef(null)
  const result = answer?.key === queryKey ? answer.data : null

  useEffect(() => {
    let active = true
    api('voice-languages').then((languages) => {
      if (active) setVoiceLanguages(languages)
    }).catch(() => {})
    return () => { active = false }
  }, [])

  const { record, recording, opening } = useRecording({
    active,
    maxSeconds: 60,
    onFile: transcribe,
    onError: setError,
  })
  const inputBusy = Boolean(busy) || opening

  async function transcribe(file) {
    setBusy('Transcribing…')
    setError('')
    try {
      const form = fileForm(voiceLanguage ? await toWav(file) : file)
      if (voiceLanguage) form.append('language', voiceLanguage)
      const transcript = await api('transcribe', queryKey, form)
      setQuestion(transcript.text.slice(0, 2000))
      setTranscriptLanguage(transcript.language ?? null)
    } catch (failure) {
      setError(failure.message)
    } finally {
      setBusy('')
    }
  }

  async function ask(event) {
    event.preventDefault()
    setBusy('Finding an answer…')
    setError('')
    setAnswer(null)
    try {
      setAnswer({ key: queryKey, data: await api('ask', queryKey, {
        question: question.trim(), language: transcriptLanguage,
      }) })
    } catch (failure) {
      setError(failure.message)
    } finally {
      setBusy('')
    }
  }

  const hasKey = Boolean(queryKey.trim())
  return (
    <>
      <section className="intro">
        <h1>Ask your organization</h1>
        <p>Type a question or record it in your language. Get an answer based on your organization’s documents.</p>
      </section>
      <div className="workspace question-workspace">
        <section className="card">
          <h2>Your question</h2>
          <form className="stack" onSubmit={ask}>
            <label>Organization query key<input type="password" autoComplete="off" value={queryKey} disabled={inputBusy || recording} onChange={(event) => {
              setQueryKey(event.target.value)
              setAnswer(null)
            }} placeholder="Enter your query key" required /></label>
            <label>Question<textarea value={question} onChange={(event) => {
              setQuestion(event.target.value)
              setTranscriptLanguage(null)
            }} placeholder="What would you like to know?" maxLength={2000} dir="auto" disabled={inputBusy || recording} required /></label>
            <label>Voice input language
              <select value={voiceLanguage} onChange={(event) => setVoiceLanguage(event.target.value)} disabled={inputBusy || recording}>
                <option value="">Auto-detect with Groq</option>
                <optgroup label="IndicConformer">
                  {Object.entries(voiceLanguages).map(([code, label]) => <option key={code} value={code}>{label}</option>)}
                </optgroup>
              </select>
            </label>
            <div className="actions voice-actions">
              <button type="button" className={recording ? 'stop' : 'secondary'} onClick={record} disabled={!hasKey || inputBusy}><Wave />{recording ? 'Stop recording' : 'Record voice'}</button>
              <button type="button" className="secondary" onClick={() => audioInput.current.click()} disabled={!hasKey || inputBusy || recording}>Upload audio</button>
              <input ref={audioInput} type="file" hidden accept=".flac,.mp3,.mp4,.mpeg,.mpga,.m4a,.ogg,.wav,.webm" onChange={(event) => {
                const file = event.target.files[0]
                event.target.value = ''
                if (file) void transcribe(file)
              }} />
            </div>
            <p className="muted voice-hint">Select an Indian language to transcribe locally with IndicConformer. Review the transcript before asking.</p>
            <Notice>{error}</Notice>
            {(busy || opening) && <p className="muted" role="status">{busy || 'Opening microphone…'}</p>}
            {recording && <p className="recording" role="status">Recording · Up to 60 seconds</p>}
            <button className="primary" disabled={!hasKey || !question.trim() || inputBusy || recording}>Ask question</button>
          </form>
        </section>
        <section className={`card answer-card ${result ? 'has-answer' : ''}`} aria-live="polite" aria-busy={busy === 'Finding an answer…'}>
          <h2>Answer</h2>
          {result ? (
            <>
              <p className="answer-text" dir="auto">{result.answer}</p>
              {active && <AnswerAudio key={result.audio_id ?? 'unavailable'} result={result} queryKey={queryKey} />}
              {result.sources.length > 0 && (
                <div className="source-list">
                  <h3>Sources</h3>
                  {result.sources.map((source, index) => (
                    <details key={`${source.document_id}-${index}`}>
                      <summary>{source.filename}{source.page != null && ` · Page ${source.page}`}</summary>
                      <p dir="auto">{source.excerpt}</p>
                    </details>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="empty-answer">
              <div className="wave-panel"><Wave /></div>
              <p>{busy === 'Finding an answer…' ? 'Searching the documents…' : 'Your answer and its sources will appear here.'}</p>
            </div>
          )}
        </section>
      </div>
    </>
  )
}

function getPage() {
  return window.location.pathname.replace(/\/$/, '') === '/admin' ? '/admin' : '/ask'
}

export default function App() {
  const [page, setPage] = useState(getPage)
  const [queryKey, setQueryKey] = useState('')

  useEffect(() => {
    const updatePage = () => setPage(getPage())
    window.addEventListener('popstate', updatePage)
    return () => window.removeEventListener('popstate', updatePage)
  }, [])

  function navigate(path) {
    if (window.location.pathname !== path) window.history.pushState({}, '', path)
    setPage(path)
  }

  function follow(event, path) {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return
    event.preventDefault()
    navigate(path)
  }

  return (
    <>
      <header className="site-header">
        <div className="shell header-content">
          <a href="/ask" className="brand" onClick={(event) => follow(event, '/ask')}><span className="brand-mark"><Wave /></span>Voice Q&amp;A</a>
          <nav aria-label="Pages">
            <a href="/ask" onClick={(event) => follow(event, '/ask')} aria-current={page === '/ask' ? 'page' : undefined}>Ask</a>
            <a href="/admin" onClick={(event) => follow(event, '/admin')} aria-current={page === '/admin' ? 'page' : undefined}>Admin</a>
          </nav>
        </div>
      </header>
      <main className="shell">
        <div hidden={page !== '/ask'}><QuestionPage active={page === '/ask'} queryKey={queryKey} setQueryKey={setQueryKey} /></div>
        <div hidden={page !== '/admin'}><AdminPage active={page === '/admin'} onTry={(key) => { setQueryKey(key); navigate('/ask') }} /></div>
      </main>
    </>
  )
}
