#!/usr/bin/env python3
"""
VOICE Q&A ASSISTANT  (Hindi + English)
======================================
Ask a question in Hindi or English -> the system finds the best answer from the JSON knowledge base.
->  the answer is spoken using your cloned voice.

RUN:
    python main.py          (the browser opens automatically)

PIPELINE:
    Microphone / Text -> STT (IndicConformer for Hindi, Whisper for English)
                ->  best-matching question in data/qa.json
                ->  XTTS-v2 speaks the answer using your reference voice

INSTALL (ek baar):
    pip install coqui-tts "transformers>=4.50,<5" "gradio>=5,<6" soundfile numpy onnxruntime

FILES (sab main.py ke folder me apne aap ban jate hain):
    data/qa.json                 <- aapke 10 Q&A (pehli baar sample apne aap ban jata hai, edit karo)
    data/voice/reference.wav     <- (optional) default reference voice; UI se bhi upload kar sakte ho
    output/cache/                <- bane hue answer audios (again instantly play hote hain)

qa.json FORMAT:
    [
      {
        "id": 1,
        "question": {"hi": "...", "en": "..."},
        "answer":   {"hi": "...", "en": "..."},
        "keywords": {"hi": ["..."], "en": ["..."]}      <- optional, matching behtar karta hai
      },
      ...
    ]
"""

import os
import sys

# ---------------------------------------------------------------- environment configuration
os.environ.setdefault("COQUI_TOS_AGREED", "1")              # Accept the XTTS license automatically
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # Disable telemetry
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTHONUTF8", "1")
for _stream in (sys.stdout, sys.stderr):                    # Ensure UTF-8 console output on Windows
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import hashlib
import json
import re
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

try:
    import gradio as gr
except ImportError:
    sys.exit('ERROR: gradio install nahi hai. Chalao:  pip install "gradio>=5,<6"')

BASE = Path(__file__).resolve().parent
os.chdir(BASE)

# ------------------------------------------------------------------- settings
DATA_DIR = BASE / "data"
QA_FILE = DATA_DIR / "qa.json"
DEFAULT_VOICE = DATA_DIR / "voice" / "reference.wav"
OUT_DIR = BASE / "output"
CACHE_DIR = OUT_DIR / "cache"
VOICE_DIR = OUT_DIR / "voices"

INDIC_MODEL = "ai4bharat/indic-conformer-600m-multilingual"   # Hindi speech recognition
WHISPER_EN = "openai/whisper-small.en"                        # English speech recognition (base.en = faster/lighter)
WHISPER_MULTI = "openai/whisper-small"                        # Hindi fallback if IndicConformer fails
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

STT_SR = 16000
TTS_SR = 24000
MAX_CHARS = 180             # Keep chunks short for XTTS
PAUSE_SECONDS = 0.25
MAX_REF_SECONDS = 20
MIN_REF_SECONDS = 3
TTS_TEMPERATURE = 0.65
MATCH_THRESHOLD = 0.45      # Scores below this threshold are treated as unmatched

FALLBACK = {
    "en": "Sorry, I don't have an answer for that question. Please try asking something else.",
    "hi": "माफ़ कीजिए, मेरे पास इस सवाल का जवाब नहीं है। कृपया कोई और सवाल पूछिए।",
}

MODEL_LOCK = threading.Lock()   # ek time par ek hi heavy model call (CPU safe)

# ----------------------------------------------------------- sample Q&A (10)
SAMPLE_QA = [
    {
        "id": 1,
        "question": {"en": "What is this project about?", "hi": "यह प्रोजेक्ट किस बारे में है?"},
        "answer": {
            "en": "This is a voice assistant that listens to your question in Hindi or English, finds the answer, and speaks it back in your own cloned voice.",
            "hi": "यह एक वॉइस असिस्टेंट है जो हिंदी या अंग्रेज़ी में आपका सवाल सुनता है, जवाब ढूँढता है, और आपकी ही क्लोन की हुई आवाज़ में बोलकर बताता है।",
        },
        "keywords": {"en": ["project", "about"], "hi": ["प्रोजेक्ट", "बारे"]},
    },
    {
        "id": 2,
        "question": {"en": "How does it work?", "hi": "यह कैसे काम करता है?"},
        "answer": {
            "en": "First, speech to text converts your question into text. Then the system matches it with the stored questions, and finally text to speech reads the answer in your cloned voice.",
            "hi": "सबसे पहले स्पीच टू टेक्स्ट आपके सवाल को टेक्स्ट में बदलता है। फिर सिस्टम उसे सेव किए गए सवालों से मिलाता है, और आखिर में टेक्स्ट टू स्पीच जवाब को आपकी क्लोन की हुई आवाज़ में सुनाता है।",
        },
        "keywords": {"en": ["work", "works"], "hi": ["कैसे", "काम"]},
    },
    {
        "id": 3,
        "question": {"en": "Which AI models are used?", "hi": "इसमें कौन से एआई मॉडल इस्तेमाल हुए हैं?"},
        "answer": {
            "en": "Indic Conformer and Whisper are used for speech recognition, and XTTS version 2 is used for voice cloning and speech generation.",
            "hi": "स्पीच पहचानने के लिए इंडिक कन्फॉर्मर और व्हिस्पर, और आवाज़ क्लोन करके बोलने के लिए एक्सटीटीएस वर्ज़न दो इस्तेमाल हुआ है।",
        },
        "keywords": {"en": ["models", "model", "ai"], "hi": ["मॉडल", "एआई"]},
    },
    {
        "id": 4,
        "question": {"en": "How does voice cloning work?", "hi": "वॉइस क्लोनिंग कैसे काम करती है?"},
        "answer": {
            "en": "The model studies a short recording of your voice, captures its tone and style, and then uses it while speaking any new sentence.",
            "hi": "मॉडल आपकी आवाज़ की एक छोटी रिकॉर्डिंग सुनकर उसका लहजा और अंदाज़ पहचान लेता है, और फिर किसी भी नए वाक्य को उसी आवाज़ में बोलता है।",
        },
        "keywords": {"en": ["cloning", "clone", "voice"], "hi": ["क्लोनिंग", "क्लोन", "वॉइस"]},
    },
    {
        "id": 5,
        "question": {"en": "Which languages does it support?", "hi": "यह कौन सी भाषाएँ सपोर्ट करता है?"},
        "answer": {
            "en": "Right now it supports Hindi and English for both listening and speaking.",
            "hi": "अभी यह सुनने और बोलने दोनों के लिए हिंदी और अंग्रेज़ी को सपोर्ट करता है।",
        },
        "keywords": {"en": ["language", "languages", "support"], "hi": ["भाषा", "भाषाएँ", "भाषाएं", "सपोर्ट"]},
    },
    {
        "id": 6,
        "question": {"en": "How long should the reference voice be?", "hi": "रेफरेंस आवाज़ कितनी लंबी होनी चाहिए?"},
        "answer": {
            "en": "Around ten to twenty seconds of clean speech from a single person, without background noise, gives the best result.",
            "hi": "बिना शोर के, एक ही व्यक्ति की दस से बीस सेकंड की साफ़ आवाज़ सबसे अच्छा नतीजा देती है।",
        },
        "keywords": {"en": ["reference", "long", "seconds"], "hi": ["रेफरेंस", "लंबी", "सेकंड"]},
    },
    {
        "id": 7,
        "question": {"en": "Does it need the internet?", "hi": "क्या इसे इंटरनेट की ज़रूरत पड़ती है?"},
        "answer": {
            "en": "Only the first time, to download the models. After that, everything runs offline on your own computer.",
            "hi": "सिर्फ़ पहली बार मॉडल डाउनलोड करने के लिए। उसके बाद सब कुछ आपके अपने कंप्यूटर पर ऑफलाइन चलता है।",
        },
        "keywords": {"en": ["internet", "offline", "online"], "hi": ["इंटरनेट", "ऑफलाइन", "ऑनलाइन"]},
    },
    {
        "id": 8,
        "question": {"en": "Where can this be used?", "hi": "इसका इस्तेमाल कहाँ हो सकता है?"},
        "answer": {
            "en": "It can be used in kiosks, help desks, education, and accessibility tools, where a familiar voice makes people more comfortable.",
            "hi": "इसे कियोस्क, हेल्प डेस्क, शिक्षा और एक्सेसिबिलिटी टूल्स में इस्तेमाल किया जा सकता है, जहाँ जानी-पहचानी आवाज़ लोगों को ज़्यादा सहज महसूस कराती है।",
        },
        "keywords": {"en": ["used", "use", "application", "applications"], "hi": ["इस्तेमाल", "उपयोग", "कहाँ"]},
    },
    {
        "id": 9,
        "question": {"en": "What are the limitations?", "hi": "इसकी सीमाएँ क्या हैं?"},
        "answer": {
            "en": "It can only answer the questions stored in its database, and on a CPU the voice generation is slower than real time.",
            "hi": "यह सिर्फ़ अपने डेटाबेस में रखे सवालों के जवाब दे सकता है, और सीपीयू पर आवाज़ बनने में थोड़ा समय लगता है।",
        },
        "keywords": {"en": ["limitations", "limitation", "drawbacks"], "hi": ["सीमाएँ", "सीमाएं", "कमियाँ", "कमी"]},
    },
    {
        "id": 10,
        "question": {"en": "What is the future scope?", "hi": "भविष्य में इसमें क्या सुधार हो सकते हैं?"},
        "answer": {
            "en": "In future it can support more Indian languages, answer open questions using a language model, and respond in real time on a GPU.",
            "hi": "भविष्य में इसमें और भारतीय भाषाएँ जोड़ी जा सकती हैं, भाषा मॉडल से खुले सवालों के जवाब दिए जा सकते हैं, और जीपीयू पर तुरंत जवाब मिल सकता है।",
        },
        "keywords": {"en": ["future", "scope", "improvement", "improvements"], "hi": ["भविष्य", "सुधार"]},
    },
]


# ======================================================================
# Q&A loading + matching
# ======================================================================
def load_qa():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not QA_FILE.exists():
        QA_FILE.write_text(json.dumps(SAMPLE_QA, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[info] Created sample Q&A file: {QA_FILE}  (replace it with your own Q&A entries)")
    try:
        data = json.loads(QA_FILE.read_text(encoding="utf-8-sig"))
    except Exception as e:
        sys.exit(f"ERROR: {QA_FILE} contains invalid JSON: {e}")
    if not isinstance(data, list) or not data:
        sys.exit(f"ERROR: {QA_FILE} must contain a non-empty Q&A list.")
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict) or not isinstance(item.get("question"), dict) \
                or not isinstance(item.get("answer"), dict):
            sys.exit(f'ERROR: qa.json entry #{i} must contain "question" and "answer" dictionaries (hi/en).')
        item.setdefault("id", i)
        if not isinstance(item.get("keywords"), dict):
            item["keywords"] = {}
    return data


STOPWORDS = {
    "en": {"the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "this", "that", "it", "of",
           "to", "in", "and", "or", "you", "your", "please", "tell", "me", "can", "i", "my", "what",
           "which", "be", "will"},
    "hi": {"है", "हैं", "का", "की", "के", "में", "को", "से", "और", "क्या", "यह", "ये", "वह", "पर", "कि",
           "तो", "भी", "सी", "सा", "हो", "सकता", "सकती", "सकते"},
}


def norm(text: str) -> str:
    text = text.lower().replace("\u0964", " ").replace("\u0965", " ")
    text = re.sub(r"[^\w\u0900-\u097F\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(text: str, lang: str):
    toks = text.split()
    stop = STOPWORDS.get(lang, set())
    kept = [t for t in toks if t not in stop]
    return kept or toks


def _sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    r = SequenceMatcher(None, a, b).ratio()
    return r if r >= 0.75 else 0.0


def soft_f1(q, c) -> float:
    if not q or not c:
        return 0.0
    p = sum(max(_sim(t, u) for u in c) for t in q) / len(q)
    r = sum(max(_sim(u, t) for t in q) for u in c) / len(c)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def match_score(query: str, cand: str, keywords, lang: str) -> float:
    qn, cn = norm(query), norm(cand)
    if not qn or not cn:
        return 0.0
    ratio = SequenceMatcher(None, qn, cn).ratio()
    f1 = soft_f1(content_tokens(qn, lang), content_tokens(cn, lang))
    score = 0.65 * f1 + 0.35 * ratio
    hits = sum(1 for k in keywords if norm(k) and norm(k) in qn)
    score += min(0.30, 0.15 * hits)
    return min(score, 1.0)


def find_answer(query: str, lang: str):
    """Returns (best entry or None, best score)."""
    best, best_score = None, 0.0
    for entry in QA:
        q = entry["question"].get(lang)
        a = entry["answer"].get(lang)
        if not q or not a:
            continue
        kws = entry.get("keywords", {}).get(lang, []) or []
        s = match_score(query, q, kws, lang)
        if s > best_score:
            best, best_score = entry, s
    return best, best_score


# ======================================================================
# Audio helpers
# ======================================================================
def read_audio_mono(path):
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
    except Exception:
        import librosa  # fallback for mp3 etc.
        mono, sr = librosa.load(str(path), sr=None, mono=True)
        mono = np.asarray(mono, dtype=np.float32)
    return mono.astype(np.float32), int(sr)


def load_audio_16k(path) -> np.ndarray:
    mono, sr = read_audio_mono(path)
    if sr != STT_SR:
        wav = torchaudio.functional.resample(torch.from_numpy(mono), sr, STT_SR)
        mono = wav.numpy()
    return mono.astype(np.float32)


def _load_with_soundfile(path, *args, **kwargs):
    """torchaudio.load replacement - FFmpeg/torchcodec ki zarurat nahi (Windows fix)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load_with_soundfile


def split_text(text: str, max_chars: int = MAX_CHARS):
    chunks = []
    for sentence in re.split(r"(?<=[.!?।])\s+", text.strip()):
        s = sentence.strip()
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            chunks.append(s)
    return chunks


# ======================================================================
# Speech-to-Text
# ======================================================================
class SpeechToText:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.indic = None
        self.whisper_hi = None
        self.whisper_en = None

    def _whisper(self, name):
        from transformers import pipeline
        return pipeline(
            "automatic-speech-recognition",
            model=name,
            token=False,
            device=0 if self.device == "cuda" else -1,
            chunk_length_s=30,
        )

    def load(self):
        from transformers import AutoModel

        print(f"[1/3] Hindi speech recognition is loading ({INDIC_MODEL}) ...")
        try:
            self.indic = AutoModel.from_pretrained(INDIC_MODEL, trust_remote_code=True)
            try:
                self.indic.to(self.device)
                self.indic.eval()
            except Exception:
                pass
        except Exception as e:
            print(f"   ! IndicConformer failed to load ({type(e).__name__}: {e})")
            print(f"   ! Using Whisper as the Hindi fallback ({WHISPER_MULTI}).")
            self.indic = None
            self.whisper_hi = self._whisper(WHISPER_MULTI)

        print(f"[2/3] Loading English speech recognition ({WHISPER_EN}) ...")
        self.whisper_en = self._whisper(WHISPER_EN)

    def transcribe(self, audio_path, lang: str) -> str:
        wav16 = load_audio_16k(audio_path)
        if wav16.size < STT_SR // 5:
            return ""
        if lang == "hi" and self.indic is not None:
            tensor = torch.from_numpy(wav16).unsqueeze(0).to(self.device)   # [1, time]
            with torch.no_grad():
                out = self.indic(tensor, "hi", "ctc")
            if isinstance(out, (list, tuple)):
                out = out[0]
            return str(out).strip()
        asr = self.whisper_hi if lang == "hi" else self.whisper_en
        kwargs = {"language": "hindi", "task": "transcribe"} if lang == "hi" else {}
        res = asr({"raw": wav16, "sampling_rate": STT_SR}, generate_kwargs=kwargs)
        return str(res["text"]).strip()


# ======================================================================
# Voice cloning text-to-speech
# ======================================================================
class VoiceCloner:
    def __init__(self):
        self.model = None
        self.voice = None   # dict: id, path, seconds, latent, embedding

    def load(self):
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[3/3] Loading voice cloning model (XTTS-v2, device={device}) ...")
        tts = TTS(XTTS_MODEL).to(device)
        self.model = tts.synthesizer.tts_model

    def set_voice(self, src) -> dict:
        mono, sr = read_audio_mono(src)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak < 1e-4:
            raise ValueError("Reference audio is silent.")
        idx = np.where(np.abs(mono) > 0.02 * peak)[0]
        if idx.size:
            mono = mono[idx[0]: idx[-1] + 1]
        mono = mono[: MAX_REF_SECONDS * sr]
        mono = mono / max(float(np.max(np.abs(mono))), 1e-6) * 0.9
        seconds = len(mono) / sr
        if seconds < MIN_REF_SECONDS:
            raise ValueError(f"Reference audio is only {seconds:.1f}s long. Please provide at least 6-10 seconds of clearly speech.")

        voice_id = hashlib.sha1(mono.tobytes()).hexdigest()[:10]
        VOICE_DIR.mkdir(parents=True, exist_ok=True)
        clean = VOICE_DIR / f"ref_{voice_id}.wav"
        sf.write(str(clean), mono, sr)

        with MODEL_LOCK:
            latent, embedding = self.model.get_conditioning_latents(audio_path=[str(clean)])
        self.voice = {"id": voice_id, "path": clean, "seconds": seconds,
                      "latent": latent, "embedding": embedding}
        return self.voice

    def speak(self, text: str, lang: str):
        """Returns (wav_path, from_cache)."""
        voice = self.voice
        if voice is None:
            raise RuntimeError("Set a reference voice first.")
        key = hashlib.sha1(f"{lang}|{text}".encode("utf-8")).hexdigest()[:12]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out = CACHE_DIR / f"{voice['id']}_{lang}_{key}.wav"
        if out.exists():
            return out, True

        pause = np.zeros(int(PAUSE_SECONDS * TTS_SR), dtype=np.float32)
        pieces = []
        with MODEL_LOCK:
            for chunk in split_text(text):
                with torch.inference_mode():
                    res = self.model.inference(
                        text=chunk,
                        language=lang,
                        gpt_cond_latent=voice["latent"],
                        speaker_embedding=voice["embedding"],
                        temperature=TTS_TEMPERATURE,
                        speed=1.0,
                        enable_text_splitting=False,
                    )
                pieces.append(np.asarray(res["wav"], dtype=np.float32))
                pieces.append(pause)
        audio = np.concatenate(pieces[:-1]) if pieces else np.zeros(1, dtype=np.float32)
        sf.write(str(out), audio, TTS_SR)
        return out, False


# ======================================================================
# Globals (single-user local demo)
# ======================================================================
QA = []
STT = SpeechToText()
CLONER = VoiceCloner()


# ======================================================================
# User interface handlers
# ======================================================================
def voice_status_md() -> str:
    v = CLONER.voice
    if v is None:
        return "⚠️ **No reference voice is configured.** Upload or record a reference voice above."
    return f"✅ **Reference voice ready** — {v['seconds']:.1f} sec ki reference use ho rahi hai."


def on_voice(path):
    if not path:
        yield voice_status_md()
        return
    yield "⏳ Processing the reference voice..."
    try:
        CLONER.set_voice(path)
    except Exception as e:
        yield f"❌ {e}"
        return
    yield voice_status_md()


def _require_voice():
    if CLONER.voice is None:
        raise gr.Error("Please upload or record a reference voice in Step 1 first.")


def _flow(question: str, lang: str, stt_time):
    entry, score = find_answer(question, lang)
    if entry is not None and score >= MATCH_THRESHOLD:
        answer = entry["answer"][lang]
        match_md = (f"✅ **Matched Q{entry['id']}** · confidence **{score * 100:.0f}%**\n\n"
                    f"> {entry['question'][lang]}")
    else:
        answer = FALLBACK[lang]
        match_md = f"⚠️ **No confident match found** (best score {score * 100:.0f}%) — Using the fallback response."

    yield (gr.update(), match_md, answer, None,
           "🔊 Generating the spoken answer using your reference voice... (CPU processing may take some time.)")

    t0 = time.time()
    try:
        wav_path, cached = CLONER.speak(answer, lang)
    except Exception as e:
        raise gr.Error(f"Voice generation failed: {e}")
    tts_time = time.time() - t0

    parts = []
    if stt_time is not None:
        parts.append(f"STT {stt_time:.1f}s")
    parts.append("TTS instant (cache)" if cached else f"TTS {tts_time:.1f}s")
    yield (gr.update(), gr.update(), gr.update(), str(wav_path), "✅ **Done** — " + " · ".join(parts))


def ask_voice(audio_path, lang):
    _require_voice()
    if not audio_path:
        raise gr.Error("Please record a question using the microphone first.")
    yield ("", "", "", None, "🎧 Listening to your question (speech → text)...")
    t0 = time.time()
    try:
        with MODEL_LOCK:
            question = STT.transcribe(audio_path, lang)
    except Exception as e:
        raise gr.Error(f"Speech recognition failed: {e}")
    stt_time = time.time() - t0
    if not question:
        raise gr.Error("No speech was detected. Please record your question again and speak clearlyly.")
    yield (question, "", "", None, f"📝 Question transcribed ({stt_time:.1f}s). Finding the best answer...")
    yield from _flow(question, lang, stt_time)


def ask_text(text, lang):
    _require_voice()
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter a question first.")
    yield (text, "", "", None, "🔎 Finding the best answer...")
    yield from _flow(text, lang, None)


def pregenerate(langs):
    _require_voice()
    langs = langs or []
    if not langs:
        raise gr.Error("Select at least one language.")
    jobs = []
    for lang in langs:
        for entry in QA:
            a = entry["answer"].get(lang)
            if a:
                jobs.append((lang, a))
        jobs.append((lang, FALLBACK[lang]))
    total = len(jobs)
    for i, (lang, text) in enumerate(jobs, 1):
        yield f"⏳ **{i}/{total}** generating ({lang}) — {text[:45]}..."
        try:
            CLONER.speak(text, lang)
        except Exception as e:
            raise gr.Error(f"Pre-generate fail hua: {e}")
    yield f"✅ **All {total} answers are ready!** Answers can now play immediately during the live demo."


# ======================================================================
# User interface
# ======================================================================
 
# ======================================================================
# UI
# ======================================================================
CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
footer {display: none !important;}
#hero {text-align: center; padding: 26px 16px; border-radius: 20px; color: #ffffff;
       background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
       box-shadow: 0 10px 30px rgba(79, 70, 229, .35); margin-bottom: 10px;}
#hero h1 {margin: 0 0 6px 0; font-size: 2.1rem; color: #ffffff;}
#hero p {margin: 0; opacity: .92; font-size: 1.05rem; color: #ffffff;}
.card {border: 1px solid rgba(124, 58, 237, .28) !important; border-radius: 18px !important;
       padding: 16px !important;}
#answer-box textarea {font-size: 1.12rem !important; line-height: 1.6 !important;}
"""
 
HERO_HTML = """
<div id="hero">
  <h1>🎙️ Voice Q&amp;A Assistant</h1>
  <p>Hindi ya English me sawal poochho — jawab milega <b>aapki apni awaaz</b> me</p>
</div>
"""

LANG_CHOICES = [("हिंदी (Hindi)", "hi"), ("English", "en")]


def qa_table_md() -> str:
    rows = ["| # | हिंदी | English |", "|---|---|---|"]
    for e in QA:
        rows.append(f"| {e['id']} | {e['question'].get('hi', '-')} | {e['question'].get('en', '-')} |")
    return "\n".join(rows)


def build_ui():
    with gr.Blocks(title="Voice Q&A Assistant",
                  theme=gr.themes.Base(
                        primary_hue="indigo",
                        secondary_hue="slate",
                        neutral_hue="slate"
                    ),
                   css=CSS) as demo:
        gr.HTML(HERO_HTML)

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes="card"):
                gr.Markdown("### 🎙️ Step 1 — Configure Reference Voice")
                voice_in = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                    label="Reference voice (10–20 seconds, clearly speech, minimal background noise)")
                voice_status = gr.Markdown(voice_status_md())
                with gr.Accordion("⚡ Pre-generate answers before your presentation", open=False):
                    gr.Markdown("Voice generation can take time on a CPU. Pre-generate the responses once — "
                                "then answers will play **immediately** during the live demo.")
                    pre_langs = gr.CheckboxGroup(choices=LANG_CHOICES, value=["hi", "en"],
                                                 label="Languages to pre-generate")
                    pre_btn = gr.Button("⚡ Pre-generate Answers")
                    pre_status = gr.Markdown()

            with gr.Column(scale=6, elem_classes="card"):
                gr.Markdown("### 💬 Step 2 — Ask a Question")
                lang_in = gr.Radio(LANG_CHOICES, value="hi", label="Question language")
                mic_in = gr.Audio(sources=["microphone"], type="filepath",
                                  label="🎤 Record your question")
                ask_btn = gr.Button("🎤 Ask by Voice", variant="primary", size="lg")
                with gr.Row():
                    text_in = gr.Textbox(placeholder="Or type your question here and press Enter...",
                                         show_label=False, lines=1, scale=5)
                    text_btn = gr.Button("⌨️ Ask by Text", scale=2)

        with gr.Column(elem_classes="card"):
            gr.Markdown("### 🔊 Response")
            heard = gr.Textbox(label="📝 Your question", interactive=False)
            matched = gr.Markdown()
            answer = gr.Textbox(label="💡 Answer", lines=4, interactive=False, elem_id="answer-box")
            answer_audio = gr.Audio(label="🔈 Spoken answer", type="filepath",
                                    autoplay=True, interactive=False)
            status = gr.Markdown()

        with gr.Accordion("📋 Available Questions (from data/qa.json)", open=False):
            gr.Markdown(qa_table_md())

        outs = [heard, matched, answer, answer_audio, status]
        voice_in.change(on_voice, inputs=voice_in, outputs=voice_status)
        pre_btn.click(pregenerate, inputs=pre_langs, outputs=pre_status)
        ask_btn.click(ask_voice, inputs=[mic_in, lang_in], outputs=outs)
        text_btn.click(ask_text, inputs=[text_in, lang_in], outputs=outs)
        text_in.submit(ask_text, inputs=[text_in, lang_in], outputs=outs)
    return demo


# ======================================================================
# main
# ======================================================================
def main():
    global QA
    print("=" * 72)  
    print("              VOICE Q&A ASSISTANT  (STT + JSON Q&A + VOICE CLONING)")
    print("=" * 72)

    QA = load_qa()
    print(f"[info] {len(QA)} Q&A load hue: {QA_FILE}")

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.time()
    STT.load()
    CLONER.load()
    print(f"[info] All models are ready ({time.time() - t0:.0f}s).")

    if DEFAULT_VOICE.exists():
        try:
            CLONER.set_voice(DEFAULT_VOICE)
            print(f"[info] Default reference voice loaded: {DEFAULT_VOICE}")
        except Exception as e:
            print(f"[warn] Default reference voice could not be loaded: {e}")

    demo = build_ui()
    demo.queue()
    print("\nOpening the browser... (press Ctrl+C in this terminal to stop the application)\n")
    demo.launch(inbrowser=True, server_name="127.0.0.1", allowed_paths=[str(BASE)])


if __name__ == "__main__":
    main()