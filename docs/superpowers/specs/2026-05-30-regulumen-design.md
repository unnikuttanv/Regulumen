# ReguLumen — Design Spec

> Date: 2026-05-30
> Status: Approved (design); pending user review of written spec.
> A regulatory-grade RAG assistant for medical-device compliance that answers
> with verifiable clause citations, abstains rather than hallucinates, and offers
> retrieval-grounded IEC 62304 safety-class decision support.

## 1. Problem & Audience

Compliance teams and engineers in medtech repeatedly ask questions like "Does our
software change require a new risk assessment under ISO 14971?" or "What does
IEC 62304 say about SOUP for Class C software?" Today they Ctrl-F through
200-page PDFs. ReguLumen answers in natural language and shows the **exact clause**
it quotes — because in a regulated context an unverifiable answer is worse than no
answer.

## 2. Goals & Non-Goals

**Goals (v1)**
- Citation-faithful answers: every answer maps to standard → section → clause.
- Strict abstention: refuse (and say what's missing) when not grounded.
- Retrieval-grounded IEC 62304 safety-class (A/B/C) decision support with a
  mandatory disclaimer.
- An eval harness measuring retrieval precision, citation accuracy, and
  abstention correctness against a gold Q&A set.
- One-command local Docker spin-up (Phase 1); HF Space demo (Phase 2).

**Non-Goals (roadmap, explicitly deferred)**

- **Document generation.** v1 *answers* questions and *classifies* risk; it does
  not author structured deliverables (risk-management files, SOUP lists,
  traceability matrices, compliance reports). Deferred deliberately: freely
  generated multi-section regulatory documents are the hardest thing to keep
  grounded and would undercut the faithfulness differentiator. A constrained
  "grounded report export" (assembling cited answers into a memo) is the natural
  first roadmap step.
- LangGraph multi-step regulatory reasoning.
- MCP server exposing ReguLumen to other tools.
- LiteLLM Proxy for org-wide cost control.

## 3. Confirmed Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Topology | **Shared core, two thin entry points** (Approach A) | Core is testable standalone; UI never depends on API uptime |
| Deliverable | **Local Docker (Phase 1) → HF Space (Phase 2)** | De-risk locally first, then host the public demo |
| Corpus | **Focused & deep**: EU MDR full text + 2–3 key FDA/IMDRF guidances; I source it | Tight, high-quality chunking + eval; best demo-to-effort ratio |
| LLM | **Groq free tier via LiteLLM** | Free, fast, provider-swappable |
| Embeddings | **Free hosted embeddings API** (HuggingFace Inference or Jina free tier — pinned below) | Key-light, no local model download |
| Retrieval | **Hybrid: ChromaDB (dense) + BM25 (lexical)** | Pure vector misses exact clause numbers; hybrid catches semantic + lexical |
| Chunking | **Clause-aware** (section → subsection → clause), not token-count | Citations stay precise |
| Faithfulness | **Strict — abstain by default** | Unverifiable answer is worse than none |
| Risk classification | **Hybrid: transparent §4.3 severity→class table (+ risk-control downgrade rule) + grounded LLM severity extraction**; abstains if severity undeterminable | Defensible rules + grounded RAG; reuses faithfulness logic |
| Eval | **I draft ~30 gold pairs from the corpus; user reviews/corrects** | User is reviewer-of-record; judgment-call pairs flagged |

**Embeddings provider (pinned):** HuggingFace Inference API with
`BAAI/bge-large-en-v1.5` (free tier). Fallback if rate-limited: Jina
`jina-embeddings-v2-base-en` free tier. The embedder is behind an interface so
the provider is swappable without touching retrieval logic.

## 4. Architecture

```
                ┌──────────────────────────────┐
   HF Space ───▶│  Gradio UI  (imports core)    │
                └──────────────┬───────────────┘
   Docker  ────▶│  FastAPI    (imports core)    │──▶ both call ──┐
                └──────────────────────────────┘                │
                          regulumen.core                         │
   ingestion ─▶ retrieval ─▶ generation ─▶ risk  ◀──────────────┘
                     │            │
                 Chroma+BM25   Groq (LiteLLM) + hosted embeddings
```

- **`regulumen.core`** holds all logic and imports neither FastAPI nor Gradio.
- **FastAPI** (router → service → repository, Pydantic models) is a thin layer
  over the core.
- **Gradio** imports the core in-process.
- **Local Docker (Phase 1):** compose runs FastAPI + Gradio as two services.
- **HF Space (Phase 2):** Gradio + core in one process.

### Package layout

```
src/regulumen/
  core/
    ingestion/   parsers (html_eurlex, pdf_guidance), chunker, models
    retrieval/   embedder, vector_store (chroma), bm25, hybrid
    generation/  llm (litellm), prompts, faithfulness, answerer
    risk/        severity_table, classifier
    eval/        metrics, runner
    config.py    pydantic-settings
  api/           schemas, repository, service, router, main
  ui/            app.py (gradio)
tests/           mirrors src/, TDD
eval/gold/       schema + gold_qa.jsonl (user-reviewed)
corpus/          gitignored source docs + README (license note)
scripts/         ingest.py
```

## 5. Components & Data Flow

### 5.1 Ingestion + clause-aware chunking
- Per-format parsers: EUR-Lex **HTML** → clean article/annex/clause tree; FDA/IMDRF
  **PDF** → heading-heuristic structure with a **structure-confidence flag**.
- Typed models: `SourceDoc`, `Clause` (with `clause_path`), `Chunk`
  (`clause_path`, `citation_label`, text, metadata).
- **Chunk by structure, not token count:** one clause/subclause per chunk; tiny
  siblings merged up to a max size; overlong clauses split with overlap but never
  across two clause numbers. `citation_label` example: "EU MDR, Annex I §17.2".

### 5.2 Retrieval (hybrid)
- **Dense:** ChromaDB (persistent) over chunk embeddings from the hosted API.
- **Lexical:** rank-bm25 over chunk text + citation labels (so "62304 §5.1.1" is
  findable).
- **Fusion:** normalized score combination with a configurable weight
  (`hybrid_alpha`); `top_k` configurable.

### 5.3 Generation (strict faithfulness)
- LLM via LiteLLM (Groq default, swappable).
- Prompt: grounded-only, clause-cited, with an explicit abstain path.
- **Faithfulness gate (post-generation):** each claim/citation must be supported
  by retrieved chunks; if support < `abstention_threshold`, the answer is replaced
  by a **structured abstention naming the missing information**.

### 5.4 Risk classification (hybrid rules + LLM)
- `severity_table.py`: transparent §4.3 mapping — A: no injury possible;
  B: non-serious injury; C: death or serious injury — **plus the risk-control /
  segregation downgrade rule**. Pure data + a small pure function, heavily
  commented, exhaustively unit-tested. **User is reviewer-of-record.**
- `classifier.py`: retrieve relevant 62304 text; LLM **extracts harm severity +
  risk-control context** (not a class letter); table maps severity→class;
  **abstains if severity can't be established.** Returns class + §4.3 citation +
  rationale + mandatory disclaimer.

### 5.5 Eval harness (built early)
- Gold format (JSONL): `{question, expected_citations:[label...],
  answer_must_contain:[...], must_abstain:bool}`.
- Metrics: retrieval precision/recall@k, citation accuracy, abstention
  correctness.
- **I draft ~30 pairs from the sourced corpus; user reviews/corrects**; I flag the
  judgment-call ones. Runner prints a report and writes `eval/results/<ts>.json`.
- Built right after retrieval so chunking + fusion are tuned against measured
  numbers, not guesses.

## 6. Error Handling & Edge Cases
- **Unparseable / low-confidence source structure:** flagged in chunk metadata;
  README is honest about which sources are well-structured.
- **Embeddings API rate-limit / outage:** retry with backoff, then fall back to
  the secondary provider; ingestion fails loudly rather than indexing partial.
- **Retrieval returns nothing relevant:** generation abstains.
- **Vague risk description:** classifier abstains (no severity → no class).
- **LLM cites a clause not in retrieved set:** faithfulness gate forces
  abstention.

## 7. Testing Strategy
- **TDD** throughout (test first, watch fail, implement).
- Unit tests mock LLM + embeddings/network.
- `@pytest.mark.live` tests hit real providers; skipped by default.
- §4.3 table has exhaustive unit tests over every severity/control combination.
- End-to-end check: `make ingest` → `eval runner` → grounded Q cited, unanswerable
  Q abstains, risk Q grounded + disclaimer, `docker compose up` serves both.

## 8. Stack Summary

| Layer | Choice | Why |
|---|---|---|
| LLM | Groq free (LiteLLM) | Free, fast, swappable |
| Embeddings | Hosted free API (HF Inference / Jina) | Key-light, no local download |
| Vector store | ChromaDB | Simple, persistent, local |
| Lexical | rank-bm25 | Catches exact clause numbers |
| API | FastAPI (router→service→repository) | Clean layered architecture |
| UI | Gradio | Fast demo, HF Space native |
| Pkg/test | uv · ruff · pytest (TDD) | Reproducible, disciplined |
| Deploy | Docker compose → HF Space | Phase 1 local, Phase 2 hosted |

## 9. Disclaimer & IP

Ships in product **and** README: **decision support, not regulatory advice or a
verdict** — outputs must be reviewed by a qualified person. The corpus uses only
publicly available documents (EU MDR Official Journal text, FDA/IMDRF guidance,
public summaries). Copyrighted standard texts (IEC 62304 / ISO 14971 / ISO 13485)
are **not** redistributed; the README states this explicitly as a compliance-
mindset signal.

## 10. Phasing

- **Phase 1 — Local Docker:** full pipeline, eval, both services via compose.
- **Phase 2 — HF Space:** Gradio + core single-process deploy + live demo link.

## 11. Out of Scope (Roadmap)

Grounded report export (assemble cited answers into a memo) · full document
generation (risk-management file / SOUP list drafts) · LangGraph multi-step
reasoning · MCP server · LiteLLM Proxy for org-wide cost control.
