# Escrow Assistant

AI-powered escrow assistant. Provides a chat interface to answer questions about escrow accounts, shortages, surpluses, and payment changes using RAG (Retrieval-Augmented Generation).

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
- **Observability**: Arize AX tracing with OpenInference (LangGraph, LangChain, OpenAI auto-instrumentation)

## Project Structure

```
backend/
  main.py              # FastAPI app, LangGraph agent, RAG retriever
  data/
    escrow_faqs.json   # Knowledge base for RAG (23 FAQs)
    feedback.json      # User feedback storage
  .env                 # Environment variables (API keys)
  requirements.txt     # Python dependencies

frontend/
  index.html           # Escrow analysis page with chat widget

evals/
  run_eval.py          # Evaluation script with hallucination detection
  eval_test_set.json   # 100 test questions across 6 categories
  eval_results.json    # Latest eval run results
  eval_report.md       # Generated markdown report
  generate_test_data.py # Script to generate synthetic test data

docs/
  Vendor_Demo_Prep.md           # Eval summary for vendor demos
  AI_Vendor_Evaluation_Rubric.md # Vendor scoring rubric
  Escrow_FAQ_Knowledge_Base_For_RAG.md # FAQ documentation
```

## Key Files

- `backend/main.py` - Main application entry point containing:
  - Arize AX tracing initialization (must be at top, before LangChain imports)
  - `EscrowFAQRetriever` class - RAG retrieval with vector/keyword search
    - Cosine similarity threshold: 0.3 (filters off-topic queries)
    - Keyword re-ranking boost for exact matches
  - `escrow_agent()` - LangGraph node that handles questions
  - API endpoints: `/ask`, `/feedback`, `/health`
  - System prompt and conversation history handling

- `frontend/index.html` - Escrow analysis UI with:
  - Escrow analysis dashboard (sample data)
  - Floating chat widget with quick question buttons
  - Client-side FAQ cache for instant responses
  - Feedback thumbs up/down functionality

- `evals/run_eval.py` - Evaluation script with:
  - Retrieval accuracy measurement (FAQ matching)
  - Hallucination detection (dollar amounts, dates, account numbers)
  - Category-level breakdown (faq_coverage, off_topic, ambiguous, edge_case, robustness, hallucination_detection)
  - Markdown report generation

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

# Arize AX Observability (optional but recommended)
ARIZE_SPACE_ID=...                        # Your Arize Space ID
ARIZE_API_KEY=...                         # Your Arize API Key
ARIZE_PROJECT_NAME=escrow-assistant       # Project name in Arize (default: escrow-assistant)
```

## Arize AX Observability

When `ARIZE_SPACE_ID` and `ARIZE_API_KEY` are set, the app automatically sends traces to Arize AX for visualization.

**What gets traced:**
- LangGraph workflow execution (agent nodes, edges)
- LangChain operations (chains, prompts, retrievers)
- OpenAI API calls (completions, embeddings)
- Custom metadata (RAG documents retrieved, session/user IDs)

**Setup:**
1. Sign up at [arize.com](https://arize.com) and get your Space ID and API Key
2. Add the credentials to `backend/.env`
3. Restart the server - you'll see `[Arize] Tracing enabled` in logs
4. View traces in the Arize dashboard under your project name

## Running Evaluations

```powershell
# Run eval against production
backend\.venv\Scripts\python.exe evals/run_eval.py

# Run eval with LLM-as-judge scoring (requires OPENAI_API_KEY, ~$0.01)
backend\.venv\Scripts\python.exe evals/run_eval.py --judge

# Run eval against localhost (edit API_URL in run_eval.py first)
# API_URL = "http://localhost:8001/ask"
```

**Current metrics (as of Feb 2026):**
- Overall accuracy: 75%
- Hallucination detection: 100% pass rate
- Target: ≥90%

**Test categories:**
| Category | Tests | Description |
|----------|-------|-------------|
| faq_coverage | 57 | Standard FAQ matching |
| robustness | 18 | Typos, informal language |
| hallucination_detection | 10 | Checks model doesn't invent details |
| off_topic | 5 | Non-escrow queries should be rejected |
| ambiguous | 5 | Vague queries like "I have a question" |
| edge_case | 5 | Multi-intent, adversarial inputs |

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
  "id": "faq_XX",
  "category": "Category Name",
  "question": "Question text",
  "answer": "Answer text",
  "keywords": ["keyword1", "keyword2"]
}
```

### Modifying the system prompt
Edit `ESCROW_SYSTEM_PROMPT` in `backend/main.py` (around line 183).

### Adjusting retrieval threshold
Edit `VECTOR_RELEVANCE_THRESHOLD` in `EscrowFAQRetriever` class in `backend/main.py` (around line 247). Higher = stricter filtering. Current: 0.3 (cosine similarity).

### Updating frontend quick questions
Edit the `FAQ_CACHE` object in `frontend/index.html` (around line 641) for instant cached responses, or modify the quick question buttons in the HTML.

### Testing without API keys
Set `TEST_MODE=1` in `.env` to use a fake LLM that returns placeholder responses.

### Running eval after changes
```powershell
# Start local server on port 8001
backend\.venv\Scripts\python.exe -m uvicorn main:app --port 8001 --reload

# Edit evals/run_eval.py to use localhost
# API_URL = "http://localhost:8001/ask"

# Run eval
backend\.venv\Scripts\python.exe evals/run_eval.py
```

## Deployment

Configured for Render via `render.yaml`. Set environment variables in the Render dashboard.

Production URL: https://escrow-assistant.onrender.com
