# Escrow Assistant

AI-powered escrow assistant for Freedom Mortgage customers. Provides a chat interface to answer questions about escrow accounts, shortages, surpluses, and payment changes using RAG (Retrieval-Augmented Generation).

## Quick Start

```powershell
# Windows
cd backend
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# Linux/Mac
cd backend
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/ for the UI, http://localhost:8000/docs for API docs.

## Architecture

```
User Question → FastAPI Backend → RAG Retrieval (Vector Search) → LLM Response
```

- **Backend**: FastAPI + LangGraph + LangChain
- **Frontend**: Single-page HTML with Tailwind CSS, served by FastAPI at `/`
- **RAG**: OpenAI embeddings with in-memory vector store over escrow FAQs
- **LLM**: OpenAI GPT-3.5/4 or OpenRouter (configurable)
- **Observability**: Optional Arize/OpenInference tracing

## Project Structure

```
backend/
  main.py              # FastAPI app, LangGraph agent, RAG retriever
  data/
    escrow_faqs.json   # Knowledge base for RAG
    feedback.json      # User feedback storage
  .env                 # Environment variables (API keys)
  requirements.txt     # Python dependencies

frontend/
  index.html           # Escrow analysis page with chat widget
```

## Key Files

- `backend/main.py` - Main application entry point containing:
  - `EscrowFAQRetriever` class - RAG retrieval with vector/keyword search
  - `escrow_agent()` - LangGraph node that handles questions
  - API endpoints: `/ask`, `/feedback`, `/health`
  - System prompt and conversation history handling

- `frontend/index.html` - Freedom Mortgage branded UI with:
  - Escrow analysis dashboard (sample data)
  - Floating chat widget with quick question buttons
  - Client-side FAQ cache for instant responses
  - Feedback thumbs up/down functionality

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | Ask a question about escrow. Body: `{question, conversation_history?, session_id?, user_id?}` |
| POST | `/feedback` | Submit feedback. Body: `{message_id, question, answer, rating}` |
| GET | `/health` | Health check |
| GET | `/` | Serves frontend |

## Environment Variables

Required in `backend/.env`:

```bash
# Choose one LLM provider
OPENAI_API_KEY=sk-...
# OR
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini

# Optional
ENABLE_RAG=1                              # Enable/disable RAG (default: enabled)
OPENAI_EMBED_MODEL=text-embedding-3-small # Embedding model
TEST_MODE=1                               # Use fake LLM for testing

# Optional observability (Arize)
ARIZE_SPACE_ID=...
ARIZE_API_KEY=...
```

## Tech Stack

- **Python 3.10+**
- **FastAPI** - Web framework
- **LangGraph** - Agent orchestration
- **LangChain** - LLM integration, embeddings, vector store
- **OpenAI / OpenRouter** - LLM provider
- **Tailwind CSS** - Frontend styling
- **Lucide Icons** - Icon set

## Common Tasks

### Adding new FAQs
Edit `backend/data/escrow_faqs.json`. Each entry needs:
```json
{
  "id": "FAQ-XX",
  "category": "Category Name",
  "question": "Question text",
  "answer": "Answer text",
  "keywords": ["keyword1", "keyword2"]
}
```

### Modifying the system prompt
Edit `ESCROW_SYSTEM_PROMPT` in `backend/main.py` (around line 119).

### Updating frontend quick questions
Edit the `FAQ_CACHE` object in `frontend/index.html` (around line 641) for instant cached responses, or modify the quick question buttons in the HTML.

### Testing without API keys
Set `TEST_MODE=1` in `.env` to use a fake LLM that returns placeholder responses.

## Deployment

Configured for Render via `render.yaml`. Set environment variables in the Render dashboard.
