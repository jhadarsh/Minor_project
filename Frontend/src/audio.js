export async function toWav(file, maxSeconds = 65) {
  if (!file.size || file.size > 10 * 1024 * 1024) {
    throw new Error('Choose a nonempty recording up to 10 MiB.')
  }

  const AudioContext = window.AudioContext || window.webkitAudioContext
  if (!AudioContext) throw new Error('This browser cannot prepare WAV recordings.')
  const context = new AudioContext({ sampleRate: 16000 })
  try {
    let decoded
    try {
      decoded = await context.decodeAudioData(await file.arrayBuffer())
    } catch {
      throw new Error('This browser could not decode the recording. Try a WAV file.')
    }
    if (decoded.duration > maxSeconds) throw new Error(`Recordings can be up to ${maxSeconds} seconds.`)
    const channels = Array.from({ length: decoded.numberOfChannels }, (_, index) => decoded.getChannelData(index))
    const buffer = new ArrayBuffer(44 + 2 * decoded.length)
    const wav = new DataView(buffer)
    function writeTag(offset, text) {
      for (let i = 0; i < text.length; i++) wav.setUint8(offset + i, text.charCodeAt(i))
    }
    writeTag(0, 'RIFF')
    wav.setUint32(4, buffer.byteLength - 8, true)
    writeTag(8, 'WAVE')
    writeTag(12, 'fmt ')
    wav.setUint32(16, 16, true)
    wav.setUint16(20, 1, true)
    wav.setUint16(22, 1, true)
    wav.setUint32(24, decoded.sampleRate, true)
    wav.setUint32(28, decoded.sampleRate * 2, true)
    wav.setUint16(32, 2, true)
    wav.setUint16(34, 16, true)
    writeTag(36, 'data')
    wav.setUint32(40, decoded.length * 2, true)
    for (let i = 0; i < decoded.length; i++) {
      const mono = channels.reduce((sum, channel) => sum + channel[i], 0) / channels.length
      const sample = Math.max(-1, Math.min(1, mono))
      wav.setInt16(44 + 2 * i, sample < 0 ? sample * 32768 : sample * 32767, true)
    }
    return new File([buffer], 'recording.wav', { type: 'audio/wav' })
  } finally {
    await context.close()
  }
}
