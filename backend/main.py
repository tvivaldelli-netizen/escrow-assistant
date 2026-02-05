import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# =============================================================================
# Arize AX Observability - MUST be initialized BEFORE LangChain/OpenAI imports
# =============================================================================
# This ensures auto-instrumentation properly wraps all LLM calls
_TRACING = False
_tracer_provider = None

def _init_arize_tracing():
    """Initialize Arize AX tracing for LangGraph/LangChain/OpenAI observability.

    Must be called before importing LangChain or OpenAI to ensure proper
    auto-instrumentation of all LLM calls.
    """
    global _TRACING, _tracer_provider

    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")

    if not space_id or not api_key:
        print("[Arize] Tracing disabled: ARIZE_SPACE_ID and ARIZE_API_KEY not set")
        return False

    if os.getenv("TEST_MODE"):
        print("[Arize] Tracing disabled in TEST_MODE")
        return False

    try:
        from arize.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # Register tracer provider with Arize
        _tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=os.getenv("ARIZE_PROJECT_NAME", "escrow-assistant")
        )

        # Instrument LangChain (covers LangGraph and OpenAI calls made through LangChain)
        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)

        # Note: openinference-instrumentation-openai requires Python <3.14
        # LangChain instrumentor still captures OpenAI calls made via langchain-openai
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument(tracer_provider=_tracer_provider)
        except ImportError:
            pass  # Optional: only available on Python <3.14

        print(f"[Arize] Tracing enabled - project: {os.getenv('ARIZE_PROJECT_NAME', 'escrow-assistant')}")
        return True

    except ImportError as e:
        print(f"[Arize] Tracing packages not installed: {e}")
        return False
    except Exception as e:
        print(f"[Arize] Tracing initialization failed: {e}")
        return False

# Initialize tracing BEFORE importing LangChain/OpenAI
_TRACING = _init_arize_tracing()

# Import tracing utilities (with fallbacks if not available)
try:
    from openinference.instrumentation import using_prompt_template, using_metadata, using_attributes
    from opentelemetry import trace
except ImportError:
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
    trace = None

# =============================================================================
# Application imports - AFTER tracing initialization
# =============================================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time
import json
import uuid
from datetime import datetime

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
ESCROW_SYSTEM_PROMPT = """You are an Escrow Assistant. Your role is to answer questions about escrow accounts, shortages, surpluses, insurance, taxes, PMI, and payment changes.

Guidelines:
- Be helpful, clear, and concise
- Use the knowledge base FAQs provided to answer questions accurately
- If no relevant FAQs are provided (indicated by "No relevant FAQs found"), ask the customer to provide more details about their escrow question. Only suggest contacting Customer Care if the question is escrow-related but you cannot answer it.
- Never make up information about specific account details, balances, or dates
- Be empathetic - escrow can be confusing for customers
- For vague questions like "help" or "question", ask the customer to provide more details about their escrow question

Topics you can help with:
- Escrow shortages and how to pay them
- Insurance changes and refunds
- Payment changes after escrow analysis
- PMI removal requests
- Escrow balance and disbursement questions
- Refund and surplus check status
- Auto-pay and bill pay questions
- Escrow refunds after home sale

Off-topic handling:
When a question is NOT about escrow (e.g. weather, restaurants, flights, credit cards, banking, taxes filing, general life questions), respond ONLY with:
"I'm an Escrow Assistant and can only help with escrow-related questions like shortages, surpluses, insurance, taxes, PMI, and payment changes. Is there anything escrow-related I can help you with?"
Do NOT mention "Customer Care", "support", "contact us", or any referral in your off-topic response. Customer Care should only be mentioned for escrow-related questions you cannot fully answer."""


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
    - Relevance threshold filtering to avoid false positives
    - Fallback to keyword matching when embeddings unavailable
    - Graceful degradation with feature flags
    """

    # Relevance thresholds - scores below these are filtered out
    # For vector search: cosine similarity (higher = more similar, 1 = identical)
    VECTOR_RELEVANCE_THRESHOLD = 0.3  # Min similarity to consider relevant
    # For keyword search: minimum score to return results
    KEYWORD_MIN_SCORE = 3  # Require at least one keyword match

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
            List of dicts with 'content', 'metadata', and 'score' keys.
            Returns empty list if no results meet relevance threshold.
        """
        if not ENABLE_RAG or self.is_empty:
            return []

        # Skip retrieval for very short/meaningless queries (greetings, punctuation)
        clean_query = query.strip().lower()
        SKIP_QUERIES = {'?', '??', '???', 'hi', 'hello', 'hey'}
        if len(clean_query) < 3 or clean_query in SKIP_QUERIES:
            return []

        # Use vector search if available, otherwise fall back to keywords
        if not self._vectorstore:
            return self._keyword_fallback(query, k=k)

        try:
            # Use similarity_search_with_score to get relevance scores
            # Returns list of (Document, score) tuples where score is cosine similarity
            # (higher = more similar, 1.0 = identical, 0.0 = orthogonal)
            docs_with_scores = self._vectorstore.similarity_search_with_score(query, k=max(k, 5))
        except Exception:
            return self._keyword_fallback(query, k=k)

        # Filter by relevance threshold and format results
        results = []
        for doc, similarity in docs_with_scores:
            # Higher similarity = more relevant. Filter out low-relevance results.
            if similarity < self.VECTOR_RELEVANCE_THRESHOLD:
                continue

            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": similarity,
            })

            if len(results) >= k:
                break

        # Re-rank by keyword overlap to differentiate similar FAQs
        if len(results) > 1:
            query_lower = query.lower()
            for r in results:
                keywords = r["metadata"].get("keywords", [])
                for kw in keywords:
                    if kw.lower() in query_lower:
                        r["score"] += 0.1
            results.sort(key=lambda x: x["score"], reverse=True)

        # If vector search found nothing relevant, try keyword fallback
        if not results:
            return self._keyword_fallback(query, k=k)

        return results

    def _keyword_fallback(self, query: str, *, k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval when embeddings unavailable.

        This demonstrates graceful degradation for production systems.
        Applies minimum score threshold to avoid false positives on off-topic queries.
        """
        query_lower = query.lower()
        query_terms = [t for t in query_lower.split() if len(t) > 2]

        # Skip if query has no meaningful terms
        if not query_terms:
            return []

        def _score(doc: Document) -> int:
            score = 0
            content_lower = doc.page_content.lower()

            # Match keywords from metadata (strong signal)
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
                if term in content_lower:
                    score += 1

            return score

        scored_docs = [(_score(doc), doc) for doc in self._docs]
        scored_docs.sort(key=lambda item: item[0], reverse=True)

        # Apply minimum score threshold to filter out off-topic queries
        results = []
        for score, doc in scored_docs:
            if score >= self.KEYWORD_MIN_SCORE:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                })
                if len(results) >= k:
                    break

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
        if _TRACING and trace is not None:
            current_span = trace.get_current_span()
            if current_span:
                # OpenInference semantic attributes for better Arize visualization
                current_span.set_attribute("metadata.agent_type", "escrow_faq_assistant")
                current_span.set_attribute("metadata.rag_enabled", str(ENABLE_RAG))
                current_span.set_attribute("metadata.faqs_retrieved", len(retrieved_faqs))
                current_span.set_attribute("metadata.history_turns", len(conversation_history) // 2)
                current_span.set_attribute("input.value", question)
                current_span.set_attribute("retrieval.documents_count", len(retrieved_faqs))

        response = llm.invoke([
            SystemMessage(content=ESCROW_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])

        # Record output in span
        if _TRACING and trace is not None:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute("output.value", response.content[:1000] if response.content else "")

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
