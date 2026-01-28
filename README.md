# Escrow Assistant

A **production-ready AI-powered escrow assistant** for Freedom Mortgage. This application provides an intelligent chat interface to help customers understand their escrow accounts, shortages, surpluses, and payment changes.

## Features

- 🤖 **AI-Powered Chat Assistant**: Natural language Q&A about escrow accounts
- 🔍 **RAG (Retrieval-Augmented Generation)**: Vector search over curated FAQ knowledge base
- 📊 **Escrow Analysis Dashboard**: Visual display of payment details and shortages
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🎨 **Freedom Mortgage Brand**: Follows FM design system and typography

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Question                            │
│                    (escrow, shortage, PMI, etc.)                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   FastAPI Backend       │
                    │   + RAG Retrieval       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   FAQ Knowledge Base    │
                    │   (Vector Search)       │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   LLM Response          │
                    │   (OpenAI/OpenRouter)   │
                    └─────────────────────────┘
```

## Quickstart

1) Requirements
- Python 3.10+

2) Configure environment
- Copy `backend/.env.example` to `backend/.env`
- Set one LLM key: `OPENAI_API_KEY=...` or `OPENROUTER_API_KEY=...`

3) Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

4) Run
```bash
# From the root directory
./start.sh
# or
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

5) Open
- Frontend: http://localhost:8000/
- API Docs: http://localhost:8000/docs

## Project Structure
- `backend/`: FastAPI app (`main.py`), RAG retrieval, FAQ data
- `backend/data/`: Escrow FAQ knowledge base JSON
- `frontend/index.html`: Escrow analysis page with chat widget

## API Endpoints
- POST `/ask` - Ask a question about escrow
- POST `/feedback` - Submit feedback on responses
- GET `/health` - Health check

## Environment Variables
```bash
# LLM Provider (choose one)
OPENAI_API_KEY=your_openai_api_key
# OR
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini

# RAG (optional)
ENABLE_RAG=1
```

## Deploy on Render
- This repo includes `render.yaml`
- Connect your GitHub repo in Render and deploy as a Web Service
- Set environment variables in the Render dashboard

## Repository
https://github.com/tvivaldelli-netizen/escrow-assistant
