# ReguLumen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a citation-faithful regulatory RAG assistant that answers medical-device compliance questions with verifiable clause citations, abstains when ungrounded, and offers IEC 62304 safety-class decision support.

**Architecture:** A shared `regulumen.core` (ingestion → retrieval → generation → risk → eval) with thin FastAPI + Gradio entry points importing it in-process. Hybrid retrieval (ChromaDB dense + BM25 lexical) over clause-aware chunks; strict post-generation faithfulness gate; hybrid rules+LLM risk classifier.

**Tech Stack:** Python 3.13 · venv + pip · ruff · pytest (TDD) · FastAPI · Gradio · ChromaDB · rank-bm25 · LiteLLM (Groq) · HuggingFace Inference embeddings (`bge-large-en-v1.5`).

> **Tooling decision (2026-05-30):** Use Python's built-in **venv + pip + `requirements.txt`** (uv is not installed and the user chose not to add it). Python is **3.13** on this machine (plan originally said 3.11; 3.13 supersedes it). Docker is **deferred to Step 11** (not installed; revisit then). Steps 1–10 run directly via the venv's Python.

---

> **Granularity & VCS notes (read first):**
>
> - This plan is written at **task altitude**, not line-by-line TDD-with-full-code. Reason: the parser and eval code (Steps 3, 5) depend on the *actual* HTML/PDF structure of corpus documents we don't fetch until Step 2 — writing "complete code" for them now would be invented, not real. Early structural tasks (config, models) can be coded concretely; parser/eval/generation code is written test-first **just-in-time during execution**, once real documents are in hand. Each step still states its goal, files, and the test that verifies it.
> - **No git commits.** Version control is disabled per the user's instruction. The template's per-task "commit" steps are replaced by **checkpoints** (stop, run tests, report). Re-enable commits when the user turns on VCS.
>
> Executes the approved spec:
> [docs/superpowers/specs/2026-05-30-regulumen-design.md](../superpowers/specs/2026-05-30-regulumen-design.md).
> Reference the spec for rationale; this plan says what to build, in what order,
> and how to verify each step.

## Overview

Build a citation-faithful regulatory RAG assistant via TDD in vertical slices:
each step delivers a testable capability end-to-end rather than a horizontal
layer. We front-load a thin ingest→retrieve→cite path and the eval harness so
retrieval quality is *measured* while we tune, then layer strict-abstention
generation, hybrid risk classification, the FastAPI/Gradio entry points, and
Docker.

## Current State

Greenfield. Only `docs/` exists (spec + this plan). Not a git repo (per user, no
VCS for now). No code, no conventions — this plan establishes them. The corpus is
not yet sourced; I source it in Step 2.

## Target State

`docker compose up` serves FastAPI + Gradio over a shared `regulumen.core`. A
grounded question returns a clause-cited answer; an unanswerable one abstains
naming what's missing. A component description yields a §4.3-grounded IEC 62304
class + disclaimer, or abstains. `python -m regulumen.core.eval.runner` reports
retrieval precision, citation accuracy, and abstention correctness against a
user-reviewed gold set. Phase 2 (HF Space) is staged but not required for "done."

## Conventions Established Here

- Python 3.13; **`venv` + `pip` + `requirements.txt`** (one venv at `.venv/`); `ruff` lint+format; `pytest` (TDD).
- Layout per spec §4: `src/regulumen/core/{ingestion,retrieval,generation,risk,eval}`,
  `src/regulumen/{api,ui}`, `tests/` mirroring, `eval/gold/`, `corpus/` (ignored),
  `scripts/`.
- Config via `pydantic-settings`; `.env.example` committed, secrets never.
- All LLM calls behind a LiteLLM wrapper (Groq default); embeddings behind an
  interface (HF Inference `bge-large-en-v1.5` default, Jina fallback).
- Unit tests mock LLM + embeddings/network; `@pytest.mark.live` hits real
  providers, skipped by default.

---

## Steps

### Step 1: Scaffold + config + smoke (walking skeleton)

**Goal:** Installable, linted, testable empty package with typed settings.
**Files:**
- `pyproject.toml` — project metadata + setuptools src-layout config + `[tool.ruff]`
  and `[tool.pytest.ini_options]` (installed editable via `pip install -e .`).
- `requirements.txt` — pinned deps (fastapi, uvicorn, gradio, chromadb, rank-bm25,
  litellm, httpx, pydantic, pydantic-settings, pypdf, beautifulsoup4, lxml,
  pytest, ruff).
- `src/regulumen/__init__.py`, `src/regulumen/core/__init__.py`,
  `src/regulumen/core/config.py`.
- `.env.example`, `.gitignore` (ignore `.venv/`, `corpus/`, `*.chroma`, `.env`),
  `README.md` (stub).
- `tests/test_smoke.py`.

**Tooling:** create `.venv` with `python -m venv .venv`; activate; `pip install -e .`
then `pip install -r requirements.txt`. No uv, no Docker (deferred to Step 11).

**Changes:** `Settings` (pydantic-settings) with fields from spec §3/§5:
`llm_model`, `embedding_model`, `embedding_provider`, `chroma_path`,
`abstention_threshold`, `top_k`, `hybrid_alpha`, plus API-key fields.

**Test:** `pytest -q` green; `ruff check .` clean; `Settings()` loads from env.
**Expected:** Package imports; tooling works.

---

### Step 2: Source corpus + intake contract

**Goal:** Sourced IP-safe docs on disk + the typed contract everything uses.
**Files:**
- `corpus/README.md` — expected layout, provenance, license note (spec §9).
- Download script `scripts/fetch_corpus.py` — pulls EU MDR (EUR-Lex HTML) + 2–3
  FDA/IMDRF guidance PDFs (spec §3, focused & deep) into `corpus/` (gitignored).
- `src/regulumen/core/ingestion/models.py` — `SourceDoc`, `Clause`, `Chunk`
  per spec §5.1 (incl. `clause_path`, `citation_label`).
- `tests/ingestion/test_models.py`.

**Changes:** `citation_label` built from `clause_path` + doc title
(e.g. "EU MDR, Annex I §17.2"). Fetch script records source URL + retrieval date.

**Test:** Model unit tests (citation_label formatting); fetch script run manually,
verify files land + provenance recorded.
**Expected:** Real corpus present; stable typed contract downstream.

---

### Step 3: Parsers + clause-aware chunking (vertical: file → chunks)

**Goal:** Provided files become clause-accurate `Chunk`s. (Spec §5.1.)
**Files:**
- `src/regulumen/core/ingestion/parsers/{base,html_eurlex,pdf_guidance}.py` +
  `registry.py`.
- `src/regulumen/core/ingestion/chunker.py`.
- `tests/ingestion/parsers/` with small IP-safe fixture snippets; `test_chunker.py`.

**Changes:** HTML parser → article/annex/clause tree; PDF parser →
heading-heuristic + `structure_confidence` flag. Chunker: one clause/subclause per
chunk, merge tiny siblings to a max size, split overlong clauses with overlap but
**never across two clause numbers**; each chunk carries `clause_path` +
`citation_label`.

**Test:** Each parser vs fixture → expected clause boundaries; chunker asserts no
chunk spans two clause numbers and size bounds hold.
**Expected:** EU MDR yields a real clause tree; PDFs best-effort + confidence flag.

---

### Step 4: Embeddings + Chroma + hybrid retrieval (vertical: query → cited chunks)

**Goal:** A query returns the right chunks with citation labels. (Spec §5.2.)
**Files:**
- `src/regulumen/core/retrieval/embedder.py` — provider interface; HF Inference
  impl (`bge-large-en-v1.5`, query/passage prefixes), Jina fallback; retry/backoff
  (spec §6).
- `src/regulumen/core/retrieval/vector_store.py` — Chroma persistent; `index`,
  `search`.
- `src/regulumen/core/retrieval/bm25.py` — rank-bm25 over chunk text + citation
  labels.
- `src/regulumen/core/retrieval/hybrid.py` — normalized score fusion via
  `hybrid_alpha`; `retrieve(query, k)`.
- `tests/retrieval/` — embeddings mocked in CI.

**Test:** Clause-number query returns that clause even when dense misses it; a
paraphrase returns the semantically-right clause; hybrid beats either alone on
both probes.
**Expected:** Hybrid retrieval works + persists.

---

### Step 5: Eval harness + draft gold set (built EARLY to drive tuning)

**Goal:** Measure retrieval precision, citation accuracy, abstention correctness.
(Spec §5.5.)
**Files:**
- `eval/gold/schema.md`; `eval/gold/gold_qa.jsonl` — **I draft ~30 pairs from the
  sourced corpus, flag judgment-call ones; user reviews/corrects.**
- `src/regulumen/core/eval/metrics.py`, `runner.py`.
- `tests/eval/test_metrics.py`.

**Changes:** Gold format `{question, expected_citations[], answer_must_contain[],
must_abstain}`. Runner runs gold through retrieval (generation added in Step 6),
prints report, writes `eval/results/<ts>.json`.

**Test:** Synthetic gold with known answers → metrics compute correctly.
**Expected:** `python -m regulumen.core.eval.runner` prints metrics. **GATE:**
user reviews drafted gold pairs before they count as authoritative.

---

### Step 6: Eval-driven tuning pass

**Goal:** Tune chunk thresholds + `hybrid_alpha`/`top_k` against Step 5 numbers.
**Files:** config defaults + `chunker.py`/`hybrid.py` tweaks only;
`docs/eval-tuning-notes.md` (before/after — README material).

**Test:** Re-run runner; metrics improve or the trade-off is documented.
**Expected:** Defensible, measured retrieval settings.

---

### Step 7: Generation with strict faithfulness / abstention (vertical: query → answer|abstain)

**Goal:** Cited answer when grounded; structured abstention otherwise. (Spec §5.3.)
**Files:**
- `src/regulumen/core/generation/llm.py` — LiteLLM wrapper (Groq default).
- `src/regulumen/core/generation/prompts.py` — grounded-only, clause-cited, explicit
  abstain path.
- `src/regulumen/core/generation/faithfulness.py` — post-gen gate: each
  claim/citation must be supported by retrieved chunks; if support <
  `abstention_threshold`, replace with structured abstention naming what's missing.
- `src/regulumen/core/generation/answerer.py` — retrieve → generate → gate → return
  answer + citations.
- `tests/generation/` — mocked LLM.

**Test:** Mocked LLM citing a clause absent from retrieved set → gate abstains;
grounded case returns cited answer. Re-run eval runner over `must_abstain` cases.
**Expected:** Strict near-zero-hallucination behavior; citations traceable.

---

### Step 8: IEC 62304 risk classification (hybrid rules + LLM)

**Goal:** Defensible §4.3-grounded A/B/C suggestion or abstention. (Spec §5.4.)
**Files:**
- `src/regulumen/core/risk/severity_table.py` — transparent §4.3 severity→class
  table + risk-control downgrade rule; pure data + pure function, heavily
  commented. **User is reviewer-of-record.**
- `src/regulumen/core/risk/classifier.py` — retrieve 62304 text; LLM extracts harm
  severity + control context (not a class letter); table maps severity→class;
  abstains if severity undeterminable. Returns class + §4.3 citation + rationale +
  disclaimer.
- `tests/risk/test_severity_table.py` (exhaustive combinations);
  `test_classifier.py` (mocked extraction → class/abstention).

**Test:** Table covers every severity/control combo; classifier abstains on vague
input. **GATE:** user reviews the §4.3 table.
**Expected:** Grounded suggestion + mandatory disclaimer.

---

### Step 9: FastAPI service (router → service → repository)

**Goal:** Typed HTTP API over the core. (Spec §4.)
**Files:**
- `src/regulumen/api/{schemas,repository,service,router,main}.py`.
- `tests/api/test_routes.py` — TestClient + mocked service.

**Changes:** Endpoints `/ask`, `/classify`, `/health`; responses carry citations,
abstained flag, risk class + disclaimer. Service holds orchestration; router has no
business logic.

**Test:** `/ask` → answer+citations; `/classify` → class+disclaimer; `/health` ok.
**Expected:** Clean layered API, services testable without HTTP.

---

### Step 10: Gradio UI (citations + disclaimer prominent)

**Goal:** Demo surface importing the core in-process. (Spec §4.)
**Files:** `src/regulumen/ui/app.py` — Q&A tab (answer + expandable cited clauses,
clear abstention rendering) + risk tab (class + §4.3 basis + disclaimer banner).

**Test:** Thin test that UI calls the right core functions (mocked) + manual smoke.
**Expected:** Citations and disclaimer impossible to miss.

---

### Step 11: Dockerization + one-command spin-up (Phase 1 done)

**Goal:** `docker compose up` runs FastAPI + Gradio. (Spec §10 Phase 1.)
**Files:** `Dockerfile` (multi-stage), `docker-compose.yml` (api + ui; commented
LangFuse), `.dockerignore`, `Makefile`/`justfile` (`ingest`, `eval`, `up`),
`scripts/ingest.py` (parse→chunk→embed→index `corpus/`).

**Test:** Fresh `docker compose up` → `/health` ok → end-to-end question works.
**Expected:** Reproducible local demo from one command.

---

### Step 12: README + disclaimer + roadmap

**Goal:** The interview-closing README. (Spec §8/§9/§11.)
**Files:** `README.md` — problem & audience; architecture diagram; design-decisions
table (decision → alternative → why); citation-faithfulness + **real eval numbers
from Step 6**; regulatory disclaimer; one-command setup; limitations & roadmap
(grounded report export → doc generation → LangGraph → MCP → LiteLLM Proxy); IP/
licensing note.

**Test:** A new reader spins up locally following only the README.
**Expected:** Senior judgment legible on the page.

---

### Step 13 (Phase 2, optional): HF Space deploy

**Goal:** Public live demo. (Spec §10 Phase 2.)
**Files:** `app.py` entry (Gradio + core single-process), `requirements.txt` for
Spaces, `README` HF metadata header, secrets via Space settings.

**Test:** Space builds; live URL answers a grounded question + abstains correctly.
**Expected:** Live demo link for the README.

---

## Verification (end-to-end)

1. `scripts/fetch_corpus.py` + `make ingest` index the corpus without errors.
2. `python -m regulumen.core.eval.runner` reports precision/citation/abstention
   metrics; numbers recorded in README.
3. Grounded question → clause-cited answer; unanswerable → abstains naming gaps.
4. Component description → §4.3-grounded class + disclaimer; vague → abstains.
5. `docker compose up` serves UI + API; full flow works from clean checkout.

## Risks & Mitigations

- **Chunk quality varies by source format** → per-format parsers + confidence
  flags (Step 3); prefer structured EU MDR; honest README.
- **§4.3 table subtly wrong** → pure data + exhaustive tests; user review gate
  (Step 8); abstain when severity undeterminable.
- **Faithfulness gate over-abstains** → `abstention_threshold` configurable,
  measured against gold `must_abstain` cases (Steps 5–7).
- **Free embeddings API rate-limits/outage** → retry/backoff + Jina fallback;
  ingestion fails loudly, not partially (Step 4 / spec §6).
- **Gold set only as good as review** → I flag judgment-call pairs; explicit user
  review gate before they count (Step 5).

## Review Gates (need user input mid-execution)

- **Step 5:** user reviews/corrects drafted gold Q&A pairs.
- **Step 8:** user reviews the §4.3 severity→class table.
- **Step 13:** Phase 2 is opt-in.
