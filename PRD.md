# Product Requirements Document: Escrow Assistant

## Problem Statement
Mortgage customers frequently contact Customer Care with repetitive escrow questions — shortages, payment changes, surplus refunds, PMI removal, and insurance updates. These calls are costly (~$8-12/call), create long hold times, and don't require account-specific actions. Analysis of 80,000+ customer messages shows that **70%+ of escrow inquiries can be answered with standardized FAQ content**.

## Target Users
- **Primary:** Homeowners with escrowed mortgage loans who receive an annual escrow analysis statement
- **Secondary:** Customer Care agents who can use the assistant as a deflection tool or co-pilot

## Product Vision
A self-service AI chat assistant embedded in the mortgage servicing portal that answers escrow questions instantly using a curated knowledge base, reducing call volume while improving customer satisfaction.

## Key Features

| Priority | Feature | Description |
|----------|---------|-------------|
| P0 | **RAG-powered Q&A** | Retrieves relevant FAQs via vector search and generates natural-language answers grounded in the knowledge base |
| P0 | **Escrow analysis context** | Embedded on the escrow analysis page so customers get help in the moment they have questions |
| P0 | **Quick-start questions** | Pre-built buttons for the top 5 question categories (shortages, payment changes, PMI, surplus, insurance) with cached instant responses |
| P0 | **Off-topic guardrails** | Rejects non-escrow questions; never fabricates account-specific details |
| P1 | **Conversation history** | Multi-turn context (up to 3 prior turns) so customers can ask follow-ups naturally |
| P1 | **Feedback collection** | Thumbs up/down on each response, stored for quality monitoring |
| P2 | **Session & user tracking** | Optional session/user IDs for analytics and tracing |
| P2 | **Observability** | OpenTelemetry tracing via Arize for latency, retrieval quality, and LLM monitoring |

## Architecture (Current)

```
┌──────────────┐     ┌──────────────────────────────────┐
│  Frontend    │     │  FastAPI Backend                  │
│  (HTML +     │────>│                                   │
│   Tailwind)  │     │  /ask ──> EscrowFAQRetriever     │
│              │<────│          (vector + keyword search) │
│  FAQ cache   │     │       ──> LangGraph agent         │
│  for top 5   │     │       ──> LLM (OpenAI/OpenRouter) │
└──────────────┘     └──────────────────────────────────┘
                              │
                     ┌────────v────────┐
                     │ escrow_faqs.json │
                     │ (23 FAQs, embed- │
                     │  ded at startup) │
                     └─────────────────┘
```

- **Frontend:** Single-page HTML served by FastAPI at `/`, client-side FAQ cache for instant responses on top questions
- **Backend:** FastAPI + LangGraph + LangChain, in-memory vector store (OpenAI embeddings), relevance threshold filtering
- **LLM:** Configurable — OpenAI GPT-3.5/4 or OpenRouter; fake LLM available for testing (`TEST_MODE=1`)
- **Knowledge base:** 23 curated FAQs in JSON with category, keywords, and structured answers

## Success Metrics

| Metric | Target |
|--------|--------|
| Containment rate (questions resolved without calling Care) | >60% |
| FAQ retrieval accuracy (correct FAQ in top-3 results) | >85% |
| User satisfaction (thumbs up rate) | >75% |
| Avg response latency (p95) | <3s |
| Escrow-related call volume reduction | 15-25% |

## Out of Scope (v1)
- Account-specific lookups (balances, payment dates, disbursement history)
- Payment processing or escrow shortage payments through the assistant
- Authentication or PII handling
- Multi-language support
- Agent handoff / live chat escalation

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM hallucination on account details | System prompt explicitly prohibits fabricating balances/dates; RAG grounds responses in FAQ content |
| Low retrieval quality on edge cases | Relevance threshold filtering returns "no results" rather than bad results; keyword fallback as safety net |
| Customer frustration with limitations | Clear messaging that this is for general info; prominent Customer Care link for account-specific help |

## Future Considerations (v2+)
- Authenticated account lookups (balance, disbursement status, payment history)
- Proactive shortage payment flow within the chat
- Agent escalation with conversation handoff
- Analytics dashboard for FAQ gap analysis based on unanswered questions
- Multi-channel deployment (SMS, mobile app, IVR deflection)
