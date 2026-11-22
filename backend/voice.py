"""Reuse the Gradio demo's voice models with organization-scoped references."""

import hashlib
import tempfile
import threading
from pathlib import Path

XTTS_LANGUAGES = frozenset("ar cs de en es fr hi hu it ja ko nl pl pt ru tr zh-cn".split())
INDIC_LANGUAGES = dict(zip(
    "as bn brx doi gu hi kn ks kok mai ml mni mr ne or pa sa sat sd ta te ur".split(),
    "Assamese Bengali Bodo Dogri Gujarati Hindi Kannada Kashmiri Konkani Maithili Malayalam Manipuri Marathi Nepali Odia Punjabi Sanskrit Santali Sindhi Tamil Telugu Urdu".split(),
))
MODEL_LOCK = threading.Lock()
_cloner = None
_assamese = None


def reference_path(storage: Path, organization_id: str) -> Path:
    return storage / "voices" / organization_id / "reference.wav"


def validate_wav(path: Path, max_seconds: int) -> None:
    import soundfile as sf

    try:
        info = sf.info(path)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Upload a readable WAV recording.") from exc
    if info.format not in {"WAV", "WAVEX"} or info.frames <= 0:
        raise ValueError("Upload a readable WAV recording.")
    if info.duration > max_seconds:
        raise ValueError(f"Recordings can be up to {max_seconds} seconds.")


def save_reference(data: bytes, destination: Path) -> float:
    import soundfile as sf

    from main import VoiceCloner

    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        original = Path(temporary) / "original.wav"
        original.write_bytes(data)
        validate_wav(original, max_seconds=60)
        samples, rate = VoiceCloner.prepare_reference(original)
        prepared = Path(temporary) / "reference.wav"
        sf.write(prepared, samples, rate)
        with MODEL_LOCK:
            prepared.replace(destination)
    return round(len(samples) / rate, 2)


def transcribe_indic(data: bytes, language: str) -> str:
    from main import STT

    with tempfile.TemporaryDirectory() as temporary:
        recording = Path(temporary) / "question.wav"
        recording.write_bytes(data)
        validate_wav(recording, max_seconds=65)
        with MODEL_LOCK:
            if STT.indic is None:
                STT.load_indic()
            return STT.transcribe(recording, language)


def synthesize(text: str, language: str, reference: Path) -> bytes:
    from main import AssameseSpeech, VoiceCloner

    global _cloner, _assamese

    with tempfile.TemporaryDirectory() as temporary, MODEL_LOCK:
        cache_dir = Path(temporary)
        if language == "as":
            if _assamese is None:
                _assamese = AssameseSpeech(cache_dir)
            _assamese.cache_dir = cache_dir
            wav, _ = _assamese.speak(text)
        else:
            reference_audio = reference.read_bytes()
            signature = hashlib.sha256(reference_audio).hexdigest()
            if _cloner is None:
                cloner = VoiceCloner(cache_dir)
                cloner.load()
                _cloner = cloner
            if _cloner.voice is None or _cloner.voice["id"] != signature:
                import soundfile as sf

                snapshot = cache_dir / "reference.wav"
                snapshot.write_bytes(reference_audio)
                _cloner.condition_voice(snapshot, signature, sf.info(snapshot).duration)
            _cloner.cache_dir = cache_dir
            wav, _ = _cloner.speak(text, language)
        return wav.read_bytes()
