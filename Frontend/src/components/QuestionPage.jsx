import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { BookOpen, FileText, Globe, Info, KeyRound, Lightbulb, MessageSquareText, Mic, Pencil, CircleDot, RotateCw, Send, Upload, Volume2, VolumeX, Zap } from 'lucide-react'
import { api, fileForm } from '../api'
import { toWav } from '../audio'
import { useRecording } from '../useRecording'
import { Button, Card, Lbl, Notice, Title, Wave, inputCls, labelCls } from './Shared'

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

  if (!result.audio_id) return result.audio_message && <p className="mt-3 flex items-center gap-1.5 text-xs text-slate-500"><VolumeX size={14} />{result.audio_message}</p>
  return (
    <div className="mt-3 grid gap-1.5 rounded-xl border border-blue-100 bg-blue-50/60 p-2.5">
      <p className="flex items-center gap-1.5 text-xs font-medium text-blue-800"><Volume2 size={14} />{result.audio_voice}</p>
      {audioUrl ? <audio className="h-9 w-full" controls autoPlay src={audioUrl} aria-label="Spoken answer" /> : error ? (
        <>
          <Notice>{error}</Notice>
          <Button className="justify-self-start !py-1.5" type="button" onClick={() => { setError(''); setAttempt(attempt + 1) }}><RotateCw size={14} />Retry audio</Button>
        </>
      ) : <p role="status" className="animate-pulse text-xs text-slate-500">Preparing spoken answer…</p>}
    </div>
  )
}

export default function QuestionPage({ queryKey, setQueryKey, active }) {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [voiceLanguages, setVoiceLanguages] = useState({})
  const [ttsLanguages, setTtsLanguages] = useState({ indic_voice_cloning: {}, xtts_fallback: [] })
  const [voiceLanguage, setVoiceLanguage] = useState('')
  const [answerLanguage, setAnswerLanguage] = useState('')
  const [transcriptLanguage, setTranscriptLanguage] = useState(null)
  const audioInput = useRef(null)
  const result = answer?.key === queryKey ? answer.data : null

  useEffect(() => {
    let mounted = true
    api('voice-languages').then((languages) => {
      if (mounted) setVoiceLanguages(languages)
    }).catch(() => {})
    api('tts-languages').then((languages) => {
      if (mounted) setTtsLanguages(languages)
    }).catch(() => {})
    return () => { mounted = false }
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
      setAnswer({
        key: queryKey,
        data: await api('ask', queryKey, {
          question: question.trim(),
          language: answerLanguage || transcriptLanguage || null,
        }),
      })
    } catch (failure) {
      setError(failure.message)
    } finally {
      setBusy('')
    }
  }

  const hasKey = Boolean(queryKey.trim())
  const finding = busy === 'Finding an answer…'
  const badges = result ? [
    result.language && { icon: Globe, text: result.language },
    result.cached && { icon: Zap, text: 'Instant cache hit' },
    result.fallback_used && { icon: FileText, text: 'Offline document fallback' },
  ].filter(Boolean) : []

  return (
    <div className="flex h-full flex-col gap-3">
      <div>
        <h1 className="flex items-center gap-2 text-xl font-extrabold tracking-tight text-slate-800 sm:text-2xl">
          <Mic className="text-blue-600" size={22} />Ask your organization
        </h1>
        <p className="text-sm text-slate-500">Type or speak a question in your language and get an answer from your documents.</p>
      </div>

      <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-2">
        <Card className="flex flex-col lg:overflow-y-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          <Title icon={MessageSquareText}>Your question</Title>
          <form className="grid gap-2.5" onSubmit={ask}>
            <label className={labelCls}><Lbl icon={KeyRound}>Organization query key</Lbl>
              <input className={inputCls} type="password" autoComplete="off" value={queryKey} disabled={inputBusy || recording} onChange={(event) => {
                setQueryKey(event.target.value)
                setAnswer(null)
              }} placeholder="Enter your query key" required />
            </label>
            <label className={labelCls}><Lbl icon={Pencil}>Question</Lbl>
              <textarea className={`${inputCls} min-h-[4.5rem] resize-none`} rows={3} value={question} onChange={(event) => {
                setQuestion(event.target.value)
                setTranscriptLanguage(null)
              }} placeholder="What would you like to know?" maxLength={2000} dir="auto" disabled={inputBusy || recording} required />
            </label>
            <div className="grid gap-2.5 sm:grid-cols-2">
              <label className={labelCls}><Lbl icon={Mic}>Voice input (STT)</Lbl>
                <select className={inputCls} value={voiceLanguage} onChange={(event) => setVoiceLanguage(event.target.value)} disabled={inputBusy || recording}>
                  <option value="">Auto-detect with Groq</option>
                  <optgroup label="IndicConformer">
                    {Object.entries(voiceLanguages).map(([code, label]) => <option key={code} value={code}>{label}</option>)}
                  </optgroup>
                </select>
              </label>
              <label className={labelCls}><Lbl icon={Volume2}>Answer voice (TTS)</Lbl>
                <select className={inputCls} value={answerLanguage} onChange={(event) => setAnswerLanguage(event.target.value)} disabled={inputBusy || recording}>
                  <option value="">Same as question (auto)</option>
                  <optgroup label="Indian Voice Cloning (Dhee-Indic-F5)">
                    {Object.entries(ttsLanguages.indic_voice_cloning || {}).map(([code, label]) => (
                      <option key={code} value={code}>{label} ({code})</option>
                    ))}
                  </optgroup>
                  {(ttsLanguages.xtts_fallback || []).length > 0 && (
                    <optgroup label="Other Languages (XTTS-v2)">
                      {ttsLanguages.xtts_fallback.map((code) => <option key={code} value={code}>{code}</option>)}
                    </optgroup>
                  )}
                </select>
              </label>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button type="button" variant={recording ? 'stop' : 'secondary'} onClick={record} disabled={!hasKey || inputBusy}>
                {recording ? <><span className="h-2 w-2 animate-ping rounded-full bg-rose-500" />Stop recording</> : <><Wave />Record voice</>}
              </Button>
              <Button type="button" onClick={() => audioInput.current.click()} disabled={!hasKey || inputBusy || recording}><Upload size={15} />Upload audio</Button>
              <input ref={audioInput} type="file" hidden accept=".flac,.mp3,.mp4,.mpeg,.mpga,.m4a,.ogg,.wav,.webm" onChange={(event) => {
                const file = event.target.files[0]
                event.target.value = ''
                if (file) void transcribe(file)
              }} />
            </div>
            <p className="flex items-start gap-1.5 text-[11px] leading-snug text-slate-400"><Info size={13} className="mt-px shrink-0" /><span>Pick an Indian language to transcribe locally with IndicConformer, and optionally a different language for the spoken answer.</span></p>
            <Notice>{error}</Notice>
            {(busy || opening) && <p className="animate-pulse text-xs font-medium text-blue-700" role="status">{busy || 'Opening microphone…'}</p>}
            {recording && <p className="flex items-center gap-1.5 rounded-xl bg-rose-50 px-3 py-2 text-xs font-medium text-rose-700" role="status"><CircleDot size={14} className="animate-pulse" />Recording · up to 60 seconds</p>}
            <Button variant="primary" className="w-full" disabled={!hasKey || !question.trim() || inputBusy || recording}><Send size={15} />Ask question</Button>
          </form>
        </Card>

        <Card delay={0.1} className="flex flex-col overflow-hidden" aria-live="polite" aria-busy={finding}>
          <Title icon={Lightbulb}>Answer</Title>
          <div className="min-h-0 flex-1 overflow-y-auto pr-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
            <AnimatePresence mode="wait">
              {result ? (
                <motion.div key="answer" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                  {badges.length > 0 && (
                    <div className="mb-2 flex flex-wrap gap-1.5" role="status">
                      {badges.map((b) => <span key={b.text} className="inline-flex items-center gap-1 rounded-full bg-blue-50 px-2.5 py-0.5 text-[11px] font-semibold text-blue-700 ring-1 ring-blue-100"><b.icon size={12} />{b.text}</span>)}
                    </div>
                  )}
                  <p className="whitespace-pre-wrap break-words text-[15px] leading-relaxed text-slate-700" dir="auto">{result.answer}</p>
                  {active && <AnswerAudio key={result.audio_id ?? 'unavailable'} result={result} queryKey={queryKey} />}
                  {result.sources.length > 0 && (
                    <div className="mt-3">
                      <h3 className="mb-1 flex items-center gap-1.5 text-sm font-bold text-slate-700"><BookOpen size={15} />Sources</h3>
                      {result.sources.map((source, index) => (
                        <details key={`${source.document_id}-${index}`} className="group border-t border-slate-100 py-1.5">
                          <summary className="cursor-pointer text-sm font-semibold text-blue-700 transition hover:text-blue-900">
                            <FileText size={14} className="-mt-0.5 mr-1 inline" />{source.filename}{source.page != null && ` · Page ${source.page}`}
                          </summary>
                          <p className="mt-1 break-words text-xs text-slate-500" dir="auto">{source.excerpt}</p>
                        </details>
                      ))}
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="grid h-full min-h-[10rem] place-content-center justify-items-center gap-3 text-center text-sm text-slate-500">
                  <div className="relative grid h-20 w-20 place-items-center">
                    {[0, 1].map((i) => (
                      <motion.span key={i} className="absolute inset-0 rounded-full border-2 border-blue-300"
                        animate={{ scale: [0.7, 1.4], opacity: [0.7, 0] }}
                        transition={{ duration: finding ? 1.2 : 2.6, repeat: Infinity, delay: i * (finding ? 0.6 : 1.3), ease: 'easeOut' }} />
                    ))}
                    <div className="grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 text-white shadow-lg shadow-blue-500/30">
                      <Wave className="scale-125" />
                    </div>
                  </div>
                  <p>{finding ? 'Searching the documents…' : 'Your answer and its sources will appear here.'}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </Card>
      </div>
    </div>
  )
}