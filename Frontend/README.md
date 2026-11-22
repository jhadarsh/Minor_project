# Voice Q&A frontend

Run the API from the repository root after setting `GROQ_API_KEY` in the root `.env`:

```bash
conda activate minor-project
python -m uvicorn backend.main:app --reload --port 8000
```

In a second terminal, run:

```bash
cd Frontend
npm install
npm run dev
```

- [Admin workspace](http://localhost:5173/admin) creates an organization using `BACKEND_ADMIN_KEY` from the root `.env`. Save the returned upload and query keys. The upload forms also accept an existing organization's upload key. Use **Organization voice** to record or upload a 10–20 second sample, preview it, and save it as the reference voice.
- [Ask page](http://localhost:5173/ask) accepts an organization query key. Type a question or record/upload audio, review the transcript, then submit it. The API returns answer text, document citations, and spoken playback when available.
- Microphone recording requires HTTPS or localhost. Choose an Indian language for local IndicConformer transcription or use Groq automatic language detection. The answer uses the organization’s reference voice in supported XTTS languages. Assamese playback uses a built-in voice.

Vite proxies API requests to port 8000 by default. Start Vite with `BACKEND_URL` set to change the target. For a production build, serve the frontend with SPA routing and proxy `/v1/*` and `/health` to FastAPI on the same origin.
