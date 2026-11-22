# Organization Q&A API

From the repository root, put your Groq API key in `.env` and keep the generated `BACKEND_ADMIN_KEY` private. The default Groq model is `openai/gpt-oss-20b`. Set `GROQ_MODEL` to use another model that supports JSON mode.

```bash
conda activate minor-project
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

Open [Swagger UI](http://127.0.0.1:8000/docs). Use **Authorize** to set the bearer key required for each step.

1. With `BACKEND_ADMIN_KEY`, call `POST /v1/organizations` with `{"name":"Example Organization"}`. Save the returned upload and query keys. The API shows them only when creating the organization.
2. With the organization's upload key, call `POST /v1/documents` and send the file in the multipart field `file`. Upload a searchable PDF or a UTF-8 TXT or Markdown transcript. The file limit is 10 MiB. Scanned PDFs require OCR before upload.
3. With the organization's query key, call `POST /v1/ask` with `{"question":"What are the office hours?"}`. The response contains plain answer text and supporting excerpts with filenames and PDF page numbers when available. Keep the query key on the organization's server when integrating with a public website.

Document text persists in `backend/storage/rag.sqlite3`. For small document collections, the backend sends all extracted text in excerpts to Groq. Larger collections use SQLite FTS5 to select excerpts for the question. Answers must cite the supplied excerpts. This phase returns text for the future voice-cloning integration.

The organization's query key also authorizes `POST /v1/transcribe`. Send an audio file in the multipart field `file` to receive the transcript as `{"text":"..."}`. Audio is sent to Groq's multilingual Whisper model, with automatic language detection. Set `GROQ_STT_MODEL` to change the model. For larger document collections, the API can rewrite search terms in the document language and retry when the first search yields no supported answer.

## Organization voice

Install the voice dependencies with `python -m pip install -r backend/requirements-voice.txt` if the Gradio demo environment is not already configured. The API loads the speech models on first use. IndicConformer requires access to AI4Bharat's gated Hugging Face repository if its files are not already cached.

- With the organization upload key, send `POST /v1/voice` as multipart with a WAV in `file` and `consent=true`. Upload 10–20 seconds of one speaker. The upload replaces the organization's reference and stays on the backend.
- For local speech recognition, send a WAV and an IndicConformer `language` code to `POST /v1/transcribe`. `GET /v1/voice-languages` lists all 22 codes. Omit the language to use Groq auto-detection.
- `POST /v1/ask` returns `audio_id` when playback is available. Use the same organization's query key to fetch a WAV from `GET /v1/answers/{audio_id}/audio` within 24 hours. XTTS clones the reference in its supported languages, including Hindi and English. Assamese uses the built-in Assamese voice. Other languages outside XTTS are returned as text.
