# DITM College Chatbot (RAG + Ollama)

A chatbot for https://www.ditmcollege.org that answers questions about admissions, courses, fees,
syllabus, seats, placements and facilities using only the college's own documents.
Everything runs locally with Ollama; no paid API is needed.

## How it works

```
data/ (PDFs + Markdown) ──ingest.py──> chunks ──nomic-embed-text──> Chroma vector DB (db/)
                                                                         │
question ──embed──> top-6 similar chunks ─────────────────────────────────┘
          ──> gemma3:4b with "answer only from this context" prompt ──> answer + sources
```

## Project layout

| Path | What it is |
|---|---|
| `data/pdfs/` | Student handbook, syllabus, AICTE approval letter, exam date sheet |
| `data/curated/` | Fees, seats/intake, placement summary, contacts (transcribed from scanned PDFs on the website) |
| `data/web/` | Text of the college website pages |
| `ingest.py` | Reads `data/`, converts tables into readable rows, splits into chunks, builds `db/` |
| `app.py` | FastAPI server: `POST /chat`, `POST /chat/stream`, chat interface at `/` |
| `static/index.html` | Full-page ChatGPT-style chat interface with suggestion cards, type-ahead suggestions and follow-up suggestions |

##TECH STACK
Layer	Technology
Language	Python (plus HTML/JavaScript for the frontend)
LLM (chat model)	gemma3:4b via Ollama (alternatives suggested: llama3.1:8b, qwen2.5:7b)
Embedding model	nomic-embed-text via Ollama
LLM runtime	Ollama, running fully locally with no paid API
Vector database	Chroma (persisted in db/)
Backend	FastAPI served with Uvicorn
API endpoints	POST /chat, POST /chat/stream (streaming), and / for the UI
Frontend	Single static index.html: a ChatGPT-style interface with suggestion cards, type-ahead and follow-up suggestions
Data sources	PDFs (handbook, syllabus, AICTE letter, date sheet), curated Markdown files (fees, seats, placements, contacts), and website page text
Ingestion	ingest.py: reads the files, converts tables to readable rows, splits them into chunks, embeds them and builds the Chroma DB

## Setup (one time)

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source venv/bin/activate
python ingest.py                     # re-run whenever files in data/ change
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 to use the chatbot.

API test:
```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"question": "What is the B.Tech fee?"}'
```

## Linking it from the college website

The chatbot is a standalone page. Host it on a server that runs Ollama (college server or VPS) and link to it
from the college website, e.g. a "Ask DITM Assistant" button pointing to `https://chat.ditmcollege.org`.
Use HTTPS (e.g. nginx + Let's Encrypt) in front of `uvicorn`.

## Editing the suggestions

The suggested questions (welcome cards, type-ahead and follow-ups) are in the `SUGGESTIONS` list
near the top of the `<script>` in `static/index.html`. Add questions that the bot answers well.

## Adding or updating knowledge

1. Drop new PDFs into `data/pdfs/`, or write/edit `.md` files in `data/curated/`.
2. Run `python ingest.py`.
3. Restart the server.

Scanned PDFs (images, no selectable text) produce no text; transcribe the important parts into a `.md`
file in `data/curated/` instead, as was done for the fee structure and placement record.

## Settings worth tuning (`app.py`)

- `CHAT_MODEL`: `gemma3:4b` is fast; `llama3.1:8b` or `qwen2.5:7b` give better answers if the machine has the RAM.
- `TOP_K`: number of chunks given to the model (default 6).
- `SYSTEM_PROMPT`: tone, language and refusal behaviour.
