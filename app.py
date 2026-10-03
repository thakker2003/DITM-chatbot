"""DITM chatbot API.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Chat UI: http://localhost:8000
"""
import difflib
import json
import re
from collections import Counter

import chromadb
import ollama
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CHAT_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 4            # fewer chunks = shorter prompt = faster first word
MAX_DISTANCE = 0.6   # cosine distance; chunks further than this are ignored
HISTORY_TURNS = 2
KEEP_ALIVE = "2h"    # keep models in memory so they don't reload after 5 idle minutes
CHAT_OPTIONS = {"temperature": 0.1, "num_predict": 250}  # cap answer length

SYSTEM_PROMPT = """You are "DITM Assistant", the official virtual help desk of Delhi Institute of Technology & Management (DITM), Ganaur, Sonipat.

Tone: professional, courteous and confident, like a well-trained admissions counsellor. Use complete, well-formed
sentences. No emojis, slang or exclamation marks. Address the user politely.

Rules:
- Answer ONLY the exact question asked, using ONLY the CONTEXT. Never invent fees, dates, seats, rules or names.
- Keep it concise. A simple question gets a precise 1-2 sentence answer. Use a bullet list only when the user asks
  for a list (e.g. courses, subjects, companies), introduced by one short sentence.
- Do NOT add extra facts the user did not ask for, and do NOT add closing lines like "let me know if you need more help".
- If the CONTEXT does not contain the answer, reply: "I'm sorry, I don't have that information at the moment.
  For accurate details, please contact the DITM admissions office at 9318389921 or admissionditm26@gmail.com."
- If the question is unrelated to DITM or college matters, politely explain that you can only assist with
  DITM-related queries such as admissions, courses, fees, syllabus, placements and campus facilities.
- Mention amounts in Rs. If the user writes in Hindi or Hinglish, reply in the same language, still professionally."""

# Greetings, thanks etc. get an instant canned reply: no document search, no model call.
SMALL_TALK = [
    (r"(hi+|hello+|hey+|hii+|namaste|namaskar|good (morning|afternoon|evening)|greetings)",
     "Hello, and welcome to DITM. I can help you with information about admissions, courses, fees, syllabus, "
     "placements and campus facilities. How may I assist you today?"),
    (r"(thanks?|thank you|thanku|thx|ty|dhanyavaad|shukriya)( so much| a lot| very much)?",
     "You're welcome. Please feel free to ask if you have any other questions about DITM."),
    (r"(ok(ay)?|cool|nice|great|good|fine|alright|got it|understood|hmm+)",
     "Certainly. Is there anything else you would like to know about DITM?"),
    (r"(bye|goodbye|see you|tata|exit)",
     "Thank you for contacting DITM. We wish you all the best."),
    (r"(how are you|how r u|kaise ho|who are you|what are you|what can you do|help)",
     "I am the DITM virtual assistant. I can provide information on courses, fees, seat intake, syllabus, "
     "admissions, placements and campus facilities. Please type your question to get started."),
]

GIBBERISH_REPLY = ("I'm sorry, I couldn't understand your message. Could you please rephrase your question? "
                   "For example: \"What is the fee for B.Tech?\" or \"Which courses does DITM offer?\"")

try:
    with open("/usr/share/dict/words") as f:
        ENGLISH = {w.strip().lower() for w in f}
except OSError:
    ENGLISH = set()
# Hinglish words and abbreviations students commonly use
EXTRA_WORDS = set("""hai hain kya ki ka ke ko kitni kitna kitne kaise kahan kab kaun konsa kon mein me mai main se
    bhi aur ya nahi nhi hoga hogi milega milegi chahiye batao bataiye bta btao sir mam maam bro pls plz thx
    btech mtech bca bba mba mca bsc diploma dpharma cse ece ee me ce aiml aids ai ml ds it lpa mdu aicte hsbte
    ditm ganaur sonipat sonepat rohtak delhi ncr haryana hostel placement placements syllabus fee fees sem
    sems semester semesters admission admissions cgpa sgpa gpa backlog backlogs reappear wifi canteen""".split())


def is_known(word: str) -> bool:
    w = word.lower()
    if w in EXTRA_WORDS or w in COMMON or w in VOCAB or w in ENGLISH:
        return True
    for suffix in ("s", "es", "ed", "ing", "ly", "er"):
        if w.endswith(suffix) and w[:-len(suffix)] in ENGLISH:
            return True
    # likely a typo of a real word (e.g. "cources")
    return len(w) >= 4 and bool(difflib.get_close_matches(w, VOCAB.keys(), n=1, cutoff=0.8))


def is_gibberish(question: str) -> bool:
    words = re.findall(r"[A-Za-z]+", question)
    if not words:
        # only numbers/symbols, e.g. "???" or "12345"
        return not re.search(r"\d", question) or len(question.strip()) < 2
    if re.search(r"(.)\1{3,}", question.lower()):          # "aaaaa", "hiiiiii" handled by small talk first
        return True
    long_words = [w for w in words if len(w) >= 3]
    if not long_words:
        return len(words) == 1 and words[0].lower() not in EXTRA_WORDS
    unknown = [w for w in long_words if not is_known(w)]
    return len(unknown) / len(long_words) > 0.5


def small_talk_reply(question: str):
    q = re.sub(r"[^\w\s]", "", question.lower()).strip()
    q = re.sub(r"\s+(there|sir|maam|bro|bot|ditm|assistant)$", "", q)
    for pattern, reply in SMALL_TALK:
        if re.fullmatch(pattern, q):
            return reply
    return None


col = chromadb.PersistentClient(path="db").get_collection("ditm")

# Every word in the indexed documents, used to fix typos in search queries ("cources" -> "courses")
VOCAB = Counter(w for doc in col.get(include=["documents"])["documents"]
                for w in re.findall(r"[a-z]{4,}", doc.lower()))
COMMON = {"what", "which", "where", "when", "does", "have", "there", "about", "tell", "your", "much",
          "many", "kitni", "kitna", "kaise", "kahan", "milega", "hain", "kaun", "konsa"}


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fix_typos(text: str) -> str:
    def fix(m):
        w = m.group(0)
        if w.lower() in VOCAB or w.lower() in COMMON:
            return w
        close = difflib.get_close_matches(w.lower(), VOCAB.keys(), n=5, cutoff=0.75)
        if not close:
            return w
        # fewest letter edits first, then same first letter (typos rarely change it),
        # then the word used most in the documents
        w = w.lower()
        return min(close, key=lambda c: (edit_distance(w, c), c[0] != w[0], -VOCAB[c]))
    return re.sub(r"[A-Za-z]{4,}", fix, text)

app = FastAPI(title="DITM Chatbot")


@app.on_event("startup")
def warm_up():
    """Load both models into memory at startup so the first user doesn't wait for it."""
    ollama.embed(model=EMBED_MODEL, input="warm up", keep_alive=KEEP_ALIVE)
    ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1}, keep_alive=KEEP_ALIVE)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


class Message(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []


def retrieve(question: str, history: list[Message]):
    # Include the previous user question so follow-ups like "and for BCA?" still retrieve well
    prev = [m.content for m in history if m.role == "user"][-1:]
    query = fix_typos(" ".join(prev + [question]))
    emb = ollama.embed(model=EMBED_MODEL, input=f"search_query: {query}", keep_alive=KEEP_ALIVE)["embeddings"][0]
    res = col.query(query_embeddings=[emb], n_results=TOP_K)
    hits = [(d, m) for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
            if dist <= MAX_DISTANCE]
    return hits


def build_messages(req: ChatRequest, hits):
    context = "\n\n---\n\n".join(d for d, _ in hits) or "(no relevant information found)"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [m.model_dump() for m in req.history[-HISTORY_TURNS * 2:]]
    messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {req.question}"})
    return messages


def sources_of(hits):
    seen, out = set(), []
    for _, m in hits:
        key = (m["source"], m["page"])
        if key not in seen:
            seen.add(key)
            out.append({"source": m["source"], "page": m["page"]})
    return out


def quick_reply(question: str):
    """Instant replies that skip the search and the model: greetings, thanks and gibberish."""
    if reply := small_talk_reply(question):
        return reply
    if is_gibberish(question):
        return GIBBERISH_REPLY
    return None


@app.post("/chat")
def chat(req: ChatRequest):
    if reply := quick_reply(req.question):
        return {"answer": reply, "sources": []}
    hits = retrieve(req.question, req.history)
    out = ollama.chat(model=CHAT_MODEL, messages=build_messages(req, hits), options=CHAT_OPTIONS,
                      keep_alive=KEEP_ALIVE)
    return {"answer": out["message"]["content"], "sources": sources_of(hits)}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Streams the answer as newline-delimited JSON: {"token": ...} lines, then {"sources": [...]}."""
    if reply := quick_reply(req.question):
        return StreamingResponse(iter([json.dumps({"token": reply}) + "\n", json.dumps({"sources": []}) + "\n"]),
                                 media_type="application/x-ndjson")
    hits = retrieve(req.question, req.history)

    def gen():
        for part in ollama.chat(model=CHAT_MODEL, messages=build_messages(req, hits),
                                options=CHAT_OPTIONS, keep_alive=KEEP_ALIVE, stream=True):
            yield json.dumps({"token": part["message"]["content"]}) + "\n"
        yield json.dumps({"sources": sources_of(hits)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/")
def home():
    return FileResponse("static/index.html")
