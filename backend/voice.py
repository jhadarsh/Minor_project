"""Voice cloning (Dhee-Indic-F5 + XTTS-v2 fallback) and IndicConformer STT."""

import hashlib
import io
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)
torch.set_num_threads(max(1, os.cpu_count() or 1))


def _load_with_soundfile(path, *args, **kwargs):
    """Replace torchaudio.load so Windows does not require FFmpeg/torchcodec."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load_with_soundfile

DHEE_INDIC_F5_REPO = "dheeyantra/dhee-indic-f5"
INDIC_CONFORMER_REPO = "ai4bharat/indic-conformer-600m-multilingual"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DHEE_NFE_STEP = int(os.getenv("DHEE_NFE_STEP", "16"))

DHEE_INDIC_F5_LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "en": "English (Indian)",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}

XTTS_LANGUAGES = frozenset("ar cs de en es fr hi hu it ja ko nl pl pt ru tr zh-cn".split())

INDIC_LANGUAGES = dict(zip(
    "as bn brx doi gu hi kn ks kok mai ml mni mr ne or pa sa sat sd ta te ur".split(),
    "Assamese Bengali Bodo Dogri Gujarati Hindi Kannada Kashmiri Konkani Maithili Malayalam Manipuri Marathi Nepali Odia Punjabi Sanskrit Santali Sindhi Tamil Telugu Urdu".split(),
))

MIN_REF_SECONDS = 3
MAX_REF_SECONDS = 10

MODEL_LOCK = threading.Lock()
_dhee_f5 = None
_dhee_vocoder = None
_indic_stt = None
_xtts = None
_xtts_sig = None
_xtts_latent = None
_xtts_embedding = None


def reference_path(storage: Path, organization_id: str) -> Path:
    return storage / "voices" / organization_id / "reference.wav"


def validate_wav(path: Path, max_seconds: int) -> None:
    try:
        info = sf.info(path)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Upload a readable WAV recording.") from exc
    if info.format not in {"WAV", "WAVEX"} or info.frames <= 0:
        raise ValueError("Upload a readable WAV recording.")
    if info.duration > max_seconds:
        raise ValueError(f"Recordings can be up to {max_seconds} seconds.")


def prepare_reference_wav(data: bytes) -> tuple[bytes, float]:
    """Validate, trim silence, cap at 10s, and normalize the reference WAV."""
    with tempfile.TemporaryDirectory() as temporary:
        original = Path(temporary) / "original.wav"
        original.write_bytes(data)
        validate_wav(original, max_seconds=60)
        samples, sr = sf.read(original, dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak < 1e-4:
        raise ValueError("Reference audio is silent.")

    active = np.where(np.abs(mono) > 0.02 * peak)[0]
    if active.size:
        mono = mono[active[0]: active[-1] + 1]

    mono = mono[: MAX_REF_SECONDS * sr]
    mono = mono / max(float(np.max(np.abs(mono))), 1e-6) * 0.9
    seconds = round(len(mono) / sr, 2)
    if seconds < MIN_REF_SECONDS:
        raise ValueError(
            f"Reference audio is only {seconds:.1f}s after trimming silence. "
            f"Provide at least {MIN_REF_SECONDS} seconds of clear speech."
        )

    buf = io.BytesIO()
    sf.write(buf, mono, sr, format="WAV")
    return buf.getvalue(), seconds


def save_reference(wav_bytes: bytes, ref_text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    txt_dest = destination.with_suffix(".txt")
    with MODEL_LOCK:
        destination.write_bytes(wav_bytes)
        txt_dest.write_text(ref_text.strip(), encoding="utf-8")
    logger.info("[VOICE] Saved reference WAV & transcript -> %s", destination)


def transcribe_indic(data: bytes, language: str) -> str:
    global _indic_stt
    from transformers import AutoModel

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary:
        recording = Path(temporary) / "question.wav"
        recording.write_bytes(data)
        validate_wav(recording, max_seconds=65)
        samples, sr = sf.read(recording, dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)

    if sr != 16000:
        mono = torchaudio.functional.resample(torch.from_numpy(mono), sr, 16000).numpy()
    if mono.size < 3200:
        return ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with MODEL_LOCK:
        if _indic_stt is None:
            logger.info("[STT] Loading IndicConformer (%s) on %s...", INDIC_CONFORMER_REPO, device)
            _indic_stt = AutoModel.from_pretrained(INDIC_CONFORMER_REPO, trust_remote_code=True)
            try:
                _indic_stt.to(device)
                _indic_stt.eval()
            except Exception:
                pass
        tensor = torch.from_numpy(mono.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = _indic_stt(tensor, language, "ctc")
        if isinstance(out, (list, tuple)):
            out = out[0]
        text = str(out).strip()

    logger.info("[STT] IndicConformer (%s) finished in %.2fs", language, time.perf_counter() - t0)
    return text


def _synthesize_dhee_f5(text: str, reference: Path) -> bytes:
    global _dhee_f5, _dhee_vocoder
    import f5_tts
    from f5_tts.infer.utils_infer import (
        infer_process,
        load_model,
        load_vocoder,
        preprocess_ref_audio_text,
    )
    from f5_tts.model import DiT
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file

    txt_path = reference.with_suffix(".txt")
    if not txt_path.is_file() or not txt_path.read_text(encoding="utf-8").strip():
        raise ValueError("Reference transcript is missing. Re-upload the reference voice via POST /v1/voice.")
    ref_text = txt_path.read_text(encoding="utf-8").strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _dhee_f5 is None or _dhee_vocoder is None:
        t_load = time.perf_counter()
        logger.info("[TTS] Loading Dhee-Indic-F5 (%s) on %s...", DHEE_INDIC_F5_REPO, device)
        repo_dir = Path(snapshot_download(repo_id=DHEE_INDIC_F5_REPO))
        ckpt_candidates = list(repo_dir.rglob("*.safetensors")) + list(repo_dir.rglob("*.pt"))
        if not ckpt_candidates:
            raise RuntimeError(f"Could not find model checkpoint in {repo_dir}")

        vocab_candidates = list(repo_dir.rglob("vocab.txt")) or list(Path(f5_tts.__file__).parent.rglob("vocab.txt"))
        vocab_file = (
            str(vocab_candidates[0])
            if vocab_candidates
            else hf_hub_download(repo_id="ai4bharat/IndicF5", filename="checkpoints/vocab.txt")
        )

        _dhee_vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
        _dhee_f5 = load_model(
            DiT,
            dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            mel_spec_type="vocos",
            vocab_file=vocab_file,
            device=device,
        )
        ckpt_path = ckpt_candidates[0]
        if ckpt_path.suffix == ".safetensors":
            state_dict = load_file(str(ckpt_path), device="cpu")
        else:
            raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            state_dict = raw.get("ema_model_state_dict", raw.get("model_state_dict", raw))

        if any(k.startswith("ema_model.") for k in state_dict):
            state_dict = {
                k[len("ema_model."):]: v
                for k, v in state_dict.items()
                if k.startswith("ema_model.")
            }
        _dhee_f5.load_state_dict(state_dict, strict=False)
        _dhee_f5.to(device).eval()
        logger.info("[TTS] Dhee-Indic-F5 loaded in %.2fs", time.perf_counter() - t_load)

    t_inf = time.perf_counter()
    ref_audio, ref_text_proc = preprocess_ref_audio_text(str(reference), ref_text)
    wav, sr, _ = infer_process(
        ref_audio,
        ref_text_proc,
        text,
        _dhee_f5,
        _dhee_vocoder,
        mel_spec_type="vocos",
        nfe_step=DHEE_NFE_STEP,
        device=device,
    )
    audio = np.asarray(wav)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=int(sr), format="WAV")
    logger.info("[TTS] Dhee-Indic-F5 synthesized %d chars in %.2fs (nfe_step=%d)", len(text), time.perf_counter() - t_inf, DHEE_NFE_STEP)
    return buf.getvalue()


def _synthesize_xtts(text: str, language: str, reference: Path) -> bytes:
    global _xtts, _xtts_sig, _xtts_latent, _xtts_embedding
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _xtts is None:
        logger.info("[TTS] Loading XTTS-v2 (%s) on %s...", XTTS_MODEL, device)
        _xtts = TTS(XTTS_MODEL).to(device).synthesizer.tts_model

    sig = hashlib.sha256(reference.read_bytes()).hexdigest()
    if _xtts_sig != sig:
        _xtts_latent, _xtts_embedding = _xtts.get_conditioning_latents(audio_path=[str(reference)])
        _xtts_sig = sig

    t_inf = time.perf_counter()
    with torch.inference_mode():
        res = _xtts.inference(
            text=text,
            language=language,
            gpt_cond_latent=_xtts_latent,
            speaker_embedding=_xtts_embedding,
            temperature=0.65,
            speed=1.0,
            enable_text_splitting=True,
        )
    buf = io.BytesIO()
    sf.write(buf, np.asarray(res["wav"], dtype=np.float32), samplerate=24000, format="WAV")
    logger.info("[TTS] XTTS-v2 (%s) synthesized %d chars in %.2fs", language, len(text), time.perf_counter() - t_inf)
    return buf.getvalue()


def synthesize(text: str, language: str, reference: Path) -> bytes:
    with MODEL_LOCK:
        if language in DHEE_INDIC_F5_LANGUAGES:
            return _synthesize_dhee_f5(text, reference)
        return _synthesize_xtts(text, language, reference)