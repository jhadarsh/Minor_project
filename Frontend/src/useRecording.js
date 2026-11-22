import { useEffect, useRef, useState } from 'react'

const extensions = {
  'audio/webm': 'webm',
  'audio/ogg': 'ogg',
  'audio/mp4': 'm4a',
  'audio/wav': 'wav',
  'video/mp4': 'mp4',
}

export function useRecording({ active, maxSeconds, onFile, onError }) {
  const [status, setStatus] = useState('')
  const session = useRef(null)

  useEffect(() => {
    if (!active) setStatus('')
    return () => {
      const current = session.current
      session.current = null
      current?.cancel()
    }
  }, [active])

  async function record() {
    if (!active) return
    if (session.current) {
      session.current.stop()
      return
    }
    onError('')
    if (!window.MediaRecorder || !navigator.mediaDevices?.getUserMedia) {
      onError('Microphone recording requires a supported browser on HTTPS or localhost. You can upload an audio file instead.')
      return
    }

    let stream
    let recorder
    let timer
    function release() {
      window.clearTimeout(timer)
      stream?.getTracks().forEach((track) => track.stop())
    }
    const current = {
      stop() {
        if (recorder && recorder.state !== 'inactive') recorder.stop()
      },
      cancel() {
        release()
        current.stop()
      },
    }
    session.current = current
    setStatus('opening')

    function finish() {
      release()
      if (session.current !== current) return false
      session.current = null
      setStatus('')
      return true
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (session.current !== current) {
        release()
        return
      }

      recorder = new MediaRecorder(stream)
      const chunks = []
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data)
      }
      recorder.onerror = () => {
        if (finish()) onError('The recording failed. Try again or upload an audio file.')
      }
      recorder.onstop = () => {
        if (!finish()) return
        if (!chunks.length) {
          onError('The recording is empty. Try again.')
          return
        }
        const mime = (recorder.mimeType || chunks[0].type || 'audio/webm').split(';')[0]
        onFile(new File(chunks, `recording.${extensions[mime] || 'webm'}`, { type: mime }))
      }
      recorder.start()
      timer = window.setTimeout(current.stop, maxSeconds * 1000)
      setStatus('recording')
    } catch {
      if (finish()) onError('Couldn’t access the microphone. Check browser permissions or upload an audio file.')
    }
  }

  return { record, recording: status === 'recording', opening: status === 'opening' }
}
