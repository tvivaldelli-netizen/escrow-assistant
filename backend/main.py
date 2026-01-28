from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import json
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Minimal observability via Arize/OpenInference (optional)
try:
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from openinference.instrumentation import using_prompt_template, using_metadata, using_attributes
    from opentelemetry import trace
    _TRACING = True
except Exception:
    def using_prompt_template(**kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_metadata(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    def using_attributes(*args, **kwargs):  # type: ignore
        from contextlib import contextmanager
        @contextmanager
        def _noop():
            yield
        return _noop()
    _TRACING = False

# LangGraph + LangChain
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore


# Data Models for Escrow Assistant
class ConversationMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class EscrowQuery(BaseModel):
    question: str
    conversation_history: Optional[List[ConversationMessage]] = []
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class EscrowResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: Optional[str] = None
    message_id: str = ""


class FeedbackRequest(BaseModel):
    message_id: str
    question: str
    answer: str
    rating: str  # 'positive' or 'negative'
    session_id: Optional[str] = None


def _init_llm():
    # Simple, test-friendly LLM init
    class _Fake:
        def __init__(self):
            pass
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            class _Msg:
                content = "I can help you with escrow questions."
                tool_calls: List[Dict[str, Any]] = []
            return _Msg()

    if os.getenv("TEST_MODE"):
        return _Fake()
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1500)
    elif os.getenv("OPENROUTER_API_KEY"):
        # Use OpenRouter via OpenAI-compatible client
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.7,
        )
    else:
        # Require a key unless running tests
        raise ValueError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY in your .env")


llm = _init_llm()


# Feature flag for RAG (enabled by default for escrow assistant)
ENABLE_RAG = os.getenv("ENABLE_RAG", "1").lower() not in {"0", "false", "no"}


# Escrow System Prompt
ESCROW_SYSTEM_PROMPT = """You are an Escrow Assistant for Freedom Mortgage customers. Your role is to answer questions about escrow accounts, shortages, surpluses, insurance, taxes, PMI, and payment changes.

Guidelines:
- Be helpful, clear, and concise
- Use the knowledge base to answer questions accurately
- If a question is outside your knowledge base, acknowledge the limitation and suggest contacting Customer Care at 1-800-220-3000
- Never make up information about specific account details, balances, or dates
- Be empathetic - escrow can be confusing for customers

Topics you can help with:
- Escrow shortages and how to pay them
- Insurance changes and refunds
- Payment changes after escrow analysis
- PMI removal requests
- Escrow balance and disbursement questions
- Refund and surplus check status"""


# RAG helper: Load escrow FAQs as LangChain documents
def _load_faq_documents(path: Path) -> List[Document]:
    """Load escrow FAQs JSON and convert to LangChain Documents."""
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []

    docs: List[Document] = []
    for faq in raw:
        question = faq.get("question", "")
        answer = faq.get("answer", "")
        if not question or not answer:
            continue

        # Combine question and answer for better retrieval
        content = f"Question: {question}\nAnswer: {answer}"

        metadata = {
            "id": faq.get("id", ""),
            "category": faq.get("category", ""),
            "keywords": faq.get("keywords", []),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


class EscrowFAQRetriever:
    """Retrieves escrow FAQs using vector similarity search.

    This class implements RAG patterns for the escrow knowledge base:
    - Vector embeddings for semantic search
    - Fallback to keyword matching when embeddings unavailable
    - Graceful degradation with feature flags
    """

    def __init__(self, data_path: Path):
        """Initialize retriever with escrow FAQ data.

        Args:
            data_path: Path to escrow_faqs.json file
        """
        self._docs = _load_faq_documents(data_path)
        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._vectorstore: Optional[InMemoryVectorStore] = None

        # Only create embeddings when RAG is enabled and we have an API key
        if ENABLE_RAG and self._docs and not os.getenv("TEST_MODE"):
            try:
                model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
                self._embeddings = OpenAIEmbeddings(model=model)
                store = InMemoryVectorStore(embedding=self._embeddings)
                store.add_documents(self._docs)
                self._vectorstore = store
            except Exception:
                # Gracefully degrade to keyword search if embeddings fail
                self._embeddings = None
                self._vectorstore = None

    @property
    def is_empty(self) -> bool:
        """Check if any documents were loaded."""
        return not self._docs

    def retrieve(self, query: str, *, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant FAQs for a question.

        Args:
            query: User's question
            k: Number of results to return

        Returns:
            List of dicts with 'content', 'metadata', and 'score' keys
        """
        if not ENABLE_RAG or self.is_empty:
            return []

        # Use vector search if available, otherwise fall back to keywords
        if not self._vectorstore:
            return self._keyword_fallback(query, k=k)

        try:
            # LangChain retriever ensures embeddings + searches are traced
            retriever = self._vectorstore.as_retriever(search_kwargs={"k": max(k, 4)})
            docs = retriever.invoke(query)
        except Exception:
            return self._keyword_fallback(query, k=k)

        # Format results with metadata and scores
        top_docs = docs[:k]
        results = []
        for doc in top_docs:
            score_val: float = 0.0
            if isinstance(doc.metadata, dict):
                maybe_score = doc.metadata.get("score")
                if isinstance(maybe_score, (int, float)):
                    score_val = float(maybe_score)
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score_val,
            })

        if not results:
            return self._keyword_fallback(query, k=k)
        return results

    def _keyword_fallback(self, query: str, *, k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval when embeddings unavailable.

        This demonstrates graceful degradation for production systems.
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        def _score(doc: Document) -> int:
            score = 0
            content_lower = doc.page_content.lower()

            # Match keywords from metadata
            keywords = doc.metadata.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 3

            # Match category
            category = doc.metadata.get("category", "").lower()
            if category and category in query_lower:
                score += 2

            # Match individual query terms in content
            for term in query_terms:
                if len(term) > 2 and term in content_lower:
                    score += 1

            return score

        scored_docs = [(_score(doc), doc) for doc in self._docs]
        scored_docs.sort(key=lambda item: item[0], reverse=True)
        top_docs = scored_docs[:k]

        results = []
        for score, doc in top_docs:
            if score > 0:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                })
        return results


# Initialize retriever at module level (loads data once at startup)
_DATA_DIR = Path(__file__).parent / "data"
FAQ_RETRIEVER = EscrowFAQRetriever(_DATA_DIR / "escrow_faqs.json")


def format_retrieved_faqs(faqs: List[Dict[str, Any]]) -> str:
    """Format retrieved FAQs as context for the LLM."""
    if not faqs:
        return "No relevant FAQs found in the knowledge base."

    lines = []
    for idx, faq in enumerate(faqs, 1):
        content = faq["content"]
        category = faq["metadata"].get("category", "General")
        lines.append(f"[{idx}] Category: {category}")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


# Escrow Agent State
class EscrowState(TypedDict):
    question: str
    conversation_history: List[dict]
    context: Optional[str]
    answer: Optional[str]
    sources: List[str]


def format_conversation_history(history: List[dict], max_turns: int = 3) -> str:
    """Format recent conversation turns for LLM context."""
    if not history:
        return ""

    recent = history[-(max_turns * 2):]  # Last N turns (user + assistant pairs)
    lines = ["Previous conversation:"]
    for msg in recent:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:500]}")  # Truncate long messages
    return "\n".join(lines)


def escrow_agent(state: EscrowState) -> EscrowState:
    """Single agent that handles escrow questions using RAG."""
    question = state["question"]
    conversation_history = state.get("conversation_history", [])

    # RAG retrieval
    retrieved_faqs = FAQ_RETRIEVER.retrieve(question, k=3)
    context = format_retrieved_faqs(retrieved_faqs)

    # Extract sources from retrieved FAQs
    sources = []
    for faq in retrieved_faqs:
        faq_id = faq["metadata"].get("id", "")
        category = faq["metadata"].get("category", "")
        if faq_id and category:
            sources.append(f"{category} ({faq_id})")

    # Format conversation history for context
    history_context = format_conversation_history(conversation_history)

    # Build prompt with context and history
    user_prompt_parts = []
    if history_context:
        user_prompt_parts.append(history_context)
    user_prompt_parts.append(f"Relevant FAQs:\n{context}")
    user_prompt_parts.append(f"Customer Question: {question}")

    user_prompt = "\n\n".join(user_prompt_parts)

    # Agent metadata and prompt template instrumentation
    with using_attributes(tags=["escrow", "faq_assistant"]):
        if _TRACING:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("metadata.agent_type", "escrow")
                current_span.set_attribute("metadata.rag_enabled", str(ENABLE_RAG))
                current_span.set_attribute("metadata.faqs_retrieved", len(retrieved_faqs))
                current_span.set_attribute("metadata.history_turns", len(conversation_history) // 2)

        response = llm.invoke([
            SystemMessage(content=ESCROW_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

    return {
        "question": question,
        "conversation_history": conversation_history,
        "context": context,
        "answer": response.content,
        "sources": sources,
    }


def build_graph():
    """Build simplified single-node workflow for escrow assistant."""
    g = StateGraph(EscrowState)
    g.add_node("escrow_agent", escrow_agent)

    # Simple linear flow
    g.add_edge(START, "escrow_agent")
    g.add_edge("escrow_agent", END)

    return g.compile()


app = FastAPI(title="Escrow Knowledge Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "frontend/index.html not found"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "escrow-knowledge-assistant"}


# Initialize tracing once at startup, not per request
if _TRACING:
    try:
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        if space_id and api_key:
            tp = register(space_id=space_id, api_key=api_key, project_name="escrow-assistant")
            LangChainInstrumentor().instrument(tracer_provider=tp, include_chains=True, include_agents=True, include_tools=True)
            LiteLLMInstrumentor().instrument(tracer_provider=tp, skip_dep_check=True)
    except Exception:
        pass


@app.post("/ask", response_model=EscrowResponse)
def ask_escrow(req: EscrowQuery):
    """Handle escrow-related questions using RAG."""
    graph = build_graph()

    # Convert conversation history to list of dicts
    history_dicts = []
    if req.conversation_history:
        for msg in req.conversation_history:
            history_dicts.append({"role": msg.role, "content": msg.content})

    # Initialize state with user question and history
    state = {
        "question": req.question,
        "conversation_history": history_dicts,
        "context": None,
        "answer": None,
        "sources": [],
    }

    # Add session and user tracking attributes to the trace
    session_id = req.session_id
    user_id = req.user_id

    attrs_kwargs = {}
    if session_id:
        attrs_kwargs["session_id"] = session_id
    if user_id:
        attrs_kwargs["user_id"] = user_id

    with using_attributes(**attrs_kwargs):
        out = graph.invoke(state)

    # Generate unique message ID for feedback tracking
    message_id = str(uuid.uuid4())[:8]

    return EscrowResponse(
        answer=out.get("answer", "I'm sorry, I couldn't process your question. Please try again."),
        sources=out.get("sources", []),
        confidence="high" if out.get("sources") else "low",
        message_id=message_id
    )


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """Record user feedback on assistant responses."""
    feedback_file = Path(__file__).parent / "data" / "feedback.json"

    # Load existing feedback
    feedback_list = []
    if feedback_file.exists():
        try:
            feedback_list = json.loads(feedback_file.read_text())
        except (json.JSONDecodeError, Exception):
            feedback_list = []

    # Append new feedback
    feedback_list.append({
        "message_id": req.message_id,
        "question": req.question,
        "answer": req.answer[:500],  # Truncate for storage
        "rating": req.rating,
        "session_id": req.session_id,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Ensure data directory exists
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    # Save
    feedback_file.write_text(json.dumps(feedback_list, indent=2))
    return {"status": "recorded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
