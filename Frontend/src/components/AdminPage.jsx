import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AudioLines, Building2, CheckCircle2, CircleDot, Copy, FileText, FileUp, KeyRound, Loader2, Mic, Paperclip, Play, Plug, Plus, Save, Search, Settings, ShieldCheck, Upload } from 'lucide-react'
import { api, fileForm } from '../api'
import { toWav } from '../audio'
import { useRecording } from '../useRecording'
import { Button, Card, Lbl, Notice, Title, Wave, inputCls, labelCls } from './Shared'

const pickerCls = 'grid gap-1 rounded-xl border border-dashed border-blue-300 bg-blue-50/50 p-2.5 text-xs font-semibold text-slate-600 transition hover:bg-blue-50'
const fileCls = 'w-full text-xs font-normal file:mr-2 file:cursor-pointer file:rounded-lg file:border file:border-blue-200 file:bg-white file:px-2.5 file:py-1 file:text-xs file:font-semibold file:text-blue-700'

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
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-x-2 gap-y-0.5">
      <label className={labelCls}>{label}<input className={`${inputCls} font-mono text-xs`} value={value} readOnly onFocus={(event) => event.target.select()} /></label>
      <Button type="button" onClick={copy}><Copy size={14} />Copy</Button>
      <span className="col-span-2 text-[11px] text-blue-700 empty:hidden" role="status">{feedback}</span>
    </div>
  )
}

function VoiceUpload({ active, uploadKey }) {
  const [file, setFile] = useState(null)
  const [refText, setRefText] = useState('')
  const [consent, setConsent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')

  const [previewUrl, setPreviewUrl] = useState('')
  const fileInput = useRef(null)
  const { record, recording, opening } = useRecording({
    active,
    maxSeconds: 10,
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
      if (refText.trim()) form.append('ref_text', refText.trim())
      const response = await api('voice', uploadKey, form)
      setStatus(`Reference voice saved · ${response.seconds}s · Transcript: "${response.ref_text}"`)
    } catch (failure) {
      setError(failure.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card delay={0.2} className="flex flex-col lg:overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
      <Title icon={Mic}>Organization voice</Title>
      <p className="mb-2 text-xs text-slate-500">Record or upload 3–10 seconds of clear speech from one person. A new recording replaces the current reference voice.</p>
      <form className="grid gap-2" onSubmit={upload}>
        <Button type="button" variant={recording ? 'stop' : 'secondary'} className="justify-self-start" onClick={record} disabled={busy || opening || !uploadKey.trim()}>
          {recording ? <><span className="h-2 w-2 animate-ping rounded-full bg-rose-500" />Stop recording</> : <><Wave />Record voice</>}
        </Button>
        {opening && <p className="animate-pulse text-xs text-slate-500" role="status">Opening microphone…</p>}
        {recording && <p className="flex items-center gap-1.5 rounded-xl bg-rose-50 px-3 py-1.5 text-xs font-medium text-rose-700" role="status"><CircleDot size={14} className="animate-pulse" />Recording · up to 10 seconds</p>}
        <label className={pickerCls}>
          <span className="flex items-center gap-1.5"><AudioLines size={14} className="text-blue-600" />Choose a reference recording</span>
          <input ref={fileInput} type="file" className={fileCls} accept="audio/*,.webm,.m4a" disabled={busy || capturing} onChange={(event) => selectFile(event.target.files[0] ?? null)} />
        </label>
        {previewUrl && !capturing && <audio className="h-9 w-full" key={previewUrl} controls src={previewUrl} aria-label="Preview reference voice" />}
        <label className={labelCls}>
          <Lbl icon={FileText}>Reference transcript (optional)</Lbl>
          <input className={inputCls} value={refText} onChange={(event) => setRefText(event.target.value)} placeholder="Leave blank to auto-transcribe" disabled={busy || capturing} />
        </label>
        <label className="flex items-start gap-2 text-xs text-slate-600">
          <input type="checkbox" className="mt-0.5 h-4 w-4 accent-blue-600" checked={consent} onChange={(event) => setConsent(event.target.checked)} disabled={busy || capturing} />
          I have permission from the speaker to clone this voice.
        </label>
        <Notice>{error}</Notice>
        {status && <p role="status" className="rounded-xl bg-emerald-50 px-3 py-2 text-xs font-medium text-emerald-700"><CheckCircle2 size={14} className="-mt-0.5 mr-1 inline" />{status}</p>}
        <Button variant="primary" disabled={busy || capturing || !file || !consent || !uploadKey.trim()}>{busy ? 'Saving…' : <><Save size={15} />Save reference voice</>}</Button>
        {!uploadKey.trim() && <p className="text-[11px] text-slate-400">Create an organization or enter its upload key first.</p>}
      </form>
    </Card>
  )
}

const endpoints = [
  ['POST /v1/documents', 'Upload key'],
  ['POST /v1/voice', 'Upload key'],
  ['POST /v1/ask', 'Query key'],
  ['POST /v1/transcribe', 'Query key'],
  ['GET /v1/answers/{audio_id}/audio', 'Query key'],
]

export default function AdminPage({ active, onTry }) {
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
    <div className="flex h-full flex-col gap-3">
      <div>
        <h1 className="flex items-center gap-2 text-xl font-extrabold tracking-tight text-slate-800 sm:text-2xl">
          <Settings className="text-blue-600" size={22} />Admin workspace
        </h1>
        <p className="text-sm text-slate-500">Upload the documents your assistant uses and manage your organization’s API keys.</p>
      </div>

      <div className="grid min-h-0 flex-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
        <Card className="flex flex-col lg:overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          <Title icon={Building2}>Organization</Title>
          <AnimatePresence mode="wait">
            {organization ? (
              <motion.div key="org" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className="grid gap-2.5">
                <p className="break-words text-base font-bold text-blue-800"><ShieldCheck size={16} className="-mt-0.5 mr-1.5 inline" />{organization.name}</p>
                <p className="text-xs text-slate-500">Save both keys now. They’re shown only when you create the organization.</p>
                <Credential key={organization.upload_key} label={<Lbl icon={KeyRound}>Upload API key</Lbl>} value={organization.upload_key} />
                <Credential key={organization.query_key} label={<Lbl icon={Search}>Query API key</Lbl>} value={organization.query_key} />
                <div className="flex flex-wrap gap-2">
                  <Button variant="primary" type="button" onClick={() => onTry(organization.query_key)}><Play size={14} />Try the assistant</Button>
                  <Button type="button" disabled={uploading} onClick={() => {
                    setOrganization(null)
                    setUploadKey('')
                    setDocuments([])
                    setName('')
                  }}><Plus size={14} />Set up another</Button>
                </div>
              </motion.div>
            ) : (
              <motion.form key="form" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }} className="grid gap-2.5" onSubmit={createOrganization}>
                <label className={labelCls}><Lbl icon={Building2}>Organization name</Lbl><input className={inputCls} required maxLength={150} value={name} onChange={(event) => setName(event.target.value)} placeholder="Enter organization name" /></label>
                <label className={labelCls}><Lbl icon={ShieldCheck}>Platform admin key</Lbl><input className={inputCls} required type="password" autoComplete="off" value={adminKey} onChange={(event) => setAdminKey(event.target.value)} placeholder="Enter admin key" /></label>
                <Notice>{createError}</Notice>
                <Button variant="primary" disabled={creating || uploading || !name.trim() || !adminKey.trim()}>{creating ? 'Creating…' : <><Plus size={15} />Create organization</>}</Button>
              </motion.form>
            )}
          </AnimatePresence>
        </Card>

        <Card delay={0.1} className="flex flex-col lg:overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          <Title icon={FileUp}>Upload a document</Title>
          <form className="grid gap-2.5" onSubmit={uploadDocument}>
            <label className={labelCls}><Lbl icon={KeyRound}>Upload API key</Lbl><input className={inputCls} required type="password" autoComplete="off" value={uploadKey} disabled={uploading} onChange={(event) => {
              setUploadKey(event.target.value)
              setDocuments([])
            }} placeholder="Enter organization upload key" /></label>
            <label className={pickerCls}>
              <span className="flex items-center gap-1.5"><Paperclip size={14} className="text-blue-600" />Choose a document</span>
              <small className="font-normal text-slate-500">Searchable PDF, TXT or Markdown · up to 10 MiB</small>
              <input ref={fileInput} type="file" className={fileCls} accept=".pdf,.txt,.md" disabled={uploading} onChange={(event) => setFile(event.target.files[0] ?? null)} />
            </label>
            <Notice>{uploadError}</Notice>
            <Button variant="primary" disabled={uploading || creating || !file || !uploadKey.trim()}>{uploading ? <><Loader2 size={15} className="animate-spin" />Indexing…</> : <><Upload size={15} />Upload document</>}</Button>
          </form>
          <ul className="mt-2 grid list-none gap-1 p-0" aria-label="Indexed documents" aria-live="polite">
            <AnimatePresence>
              {documents.map((document) => (
                <motion.li key={document.id} layout initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                  className="flex flex-wrap items-baseline justify-between gap-x-2 break-all rounded-lg bg-emerald-50 px-2.5 py-1.5 text-xs">
                  <strong className="text-slate-700"><CheckCircle2 size={13} className="-mt-0.5 mr-1 inline" />{document.filename}</strong>
                  <span className="text-emerald-700">{document.chunk_count} excerpts</span>
                </motion.li>
              ))}
            </AnimatePresence>
          </ul>
        </Card>

        <div className="min-h-0 md:col-span-2 lg:col-span-1 lg:flex lg:flex-col">
          <VoiceUpload key={uploadKey} active={active} uploadKey={uploadKey} />
        </div>
      </div>

      <Card delay={0.3} className="!p-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <p className="flex items-center gap-1.5 text-sm font-bold text-slate-700"><Plug size={15} className="text-blue-600" />Connect your platform</p>
          <div className="flex flex-wrap gap-1.5">
            {endpoints.map(([path, key]) => (
              <motion.span key={path} whileHover={{ y: -2 }} className="rounded-lg bg-slate-100 px-2 py-1 text-[11px] text-slate-600">
                <code className="font-mono text-blue-800">{path}</code> · {key}
              </motion.span>
            ))}
          </div>
        </div>
      </Card>
    </div>
  )
}