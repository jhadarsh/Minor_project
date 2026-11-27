# Organization Voice Q&A API

Multi-tenant **FastAPI** backend for document Q&A with zero-shot Indian-language voice cloning.

| Component | Technology |
|---|---|
| Voice cloning (TTS) | **Dhee-Indic-F5** (XTTS-v2 fallback) |
| Speech-to-text | **IndicConformer** (local) / **Groq Whisper** (auto-detect) |
| Retrieval | **SQLite FTS5** |
| Answer generation | **Groq LLM** with offline fallback |
| Caching | Persistent **Q&A + WAV** cache |

---

## Table of Contents

1. [Environment Variables](#1-environment-variables)
2. [Installation & Setup](#2-installation--setup)
3. [Start the Backend](#3-start-the-backend)
4. [API Endpoints & Workflow](#4-api-endpoints--workflow)
5. [Supported Languages](#5-supported-languages)
6. [Key Features for Live Demos](#6-key-features-for-live-demos)

---

## 1. Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
BACKEND_ADMIN_KEY=choose_a_strong_admin_secret
GROQ_MODEL=openai/gpt-oss-20b
GROQ_STT_MODEL=whisper-large-v3-turbo
DHEE_NFE_STEP=16
```

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key used for the Groq LLM and Whisper STT |
| `BACKEND_ADMIN_KEY` | Admin secret required to create organizations |
| `GROQ_MODEL` | Groq LLM used to generate answers |
| `GROQ_STT_MODEL` | Groq Whisper model used for auto-detect transcription |
| `DHEE_NFE_STEP` | Number of function evaluation steps for Dhee-Indic-F5 synthesis |

---

## 2. Installation & Setup

```bash
conda activate minor-project
python -m pip install -r backend/requirements.txt
huggingface-cli login
```

> **Note:** Accept the gated repository terms on [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) **before** running `huggingface-cli login`, so that `checkpoints/vocab.txt` can be downloaded on first TTS use.

---

## 3. Start the Backend

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open Swagger UI at <http://127.0.0.1:8000/docs> and use **Authorize** to set the bearer token for each step.

---

## 4. API Endpoints & Workflow

| Method & Path | Auth Key | Purpose |
|---|---|---|
| `GET /health` | None | Check server status and `GROQ_API_KEY` configuration |
| `GET /v1/voice-languages` | None | List all 22 supported IndicConformer STT languages |
| `GET /v1/tts-languages` | None | List supported TTS languages (for UI dropdowns) |
| `POST /v1/organizations` | `BACKEND_ADMIN_KEY` | Create an organization (`{"name": "..."}`); returns `upload_key` and `query_key` |
| `POST /v1/documents` | `upload_key` | Upload searchable `.pdf`, `.txt`, or `.md` (max 10 MiB, 300 pages) |
| `POST /v1/voice` | `upload_key` | Upload reference `.wav` (3–10 s, `consent=true`, optional `ref_text`) |
| `POST /v1/transcribe` | `query_key` | Transcribe audio (language set = local IndicConformer WAV; omitted = Groq Whisper auto-detect) |
| `POST /v1/ask` | `query_key` | Ask a question; optionally select a target language for cross-language answers |
| `GET /v1/answers/{audio_id}/audio` | `query_key` | Stream/download the cloned `.wav` answer (cached on disk for instant repeat playback) |

### Typical Flow

1. **Create an organization** with `POST /v1/organizations` using `BACKEND_ADMIN_KEY`.
2. **Upload documents** with `POST /v1/documents` using the `upload_key`.
3. **Upload a reference voice** with `POST /v1/voice` using the `upload_key`.
4. **(Optional) Transcribe a spoken question** with `POST /v1/transcribe` using the `query_key`.
5. **Ask a question** with `POST /v1/ask` using the `query_key`.
6. **Fetch the cloned-voice answer** from `GET /v1/answers/{audio_id}/audio`.

---

## 5. Supported Languages

### Speech-to-Text (`POST /v1/transcribe`)

**Local IndicConformer** (22 languages, WAV only):

`as`, `bn`, `brx`, `doi`, `gu`, `hi`, `kn`, `ks`, `kok`, `mai`, `ml`, `mni`, `mr`, `ne`, `or`, `pa`, `sa`, `sat`, `sd`, `ta`, `te`, `ur`

**Groq Whisper** (auto-detect):

`.wav`, `.mp3`, `.m4a`, `.webm`, `.ogg`, `.flac`, `.mp4`, `.mpeg`, `.mpga`

### TTS Voice Cloning (`POST /v1/ask` → `/v1/answers/{audio_id}/audio`)

**Dhee-Indic-F5** (12 languages):

| Code | Language | Code | Language |
|---|---|---|---|
| `as` | Assamese | `ml` | Malayalam |
| `bn` | Bengali | `mr` | Marathi |
| `en` | Indian English | `or` | Odia |
| `gu` | Gujarati | `pa` | Punjabi |
| `hi` | Hindi | `ta` | Tamil |
| `kn` | Kannada | `te` | Telugu |

**XTTS-v2 Fallback** (15 additional languages):

`ar`, `cs`, `de`, `es`, `fr`, `hu`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `ru`, `tr`, `zh-cn`

---

## 6. Key Features for Live Demos

- **Cross-Language Answers** — In `POST /v1/ask`, pick any language from the language dropdown. For example, ask in Hindi and select `bn` to get and hear the answer in Bengali.
- **Instant Cache** — Repeated `(question, target_language)` pairs return `"cached": true` immediately from `qa_cache` and play the saved WAV from `backend/storage/cache/`. Uploading new documents or a new reference voice automatically invalidates stale cache entries.
- **Groq Offline Fallback** — If Groq is unreachable or rate-limited, `/v1/ask` returns `"fallback_used": true` with the best-matching sentences extracted locally via SQLite FTS5.
