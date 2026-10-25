---
title: ReguLumen
emoji: 📑
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Citation-faithful regulatory RAG for medical-device compliance.
---

# ReguLumen

**A citation-faithful RAG assistant for medical-device compliance.** It answers
regulatory questions with verifiable clause citations, **abstains rather than
hallucinates**, and offers IEC 62304 software safety-class (A/B/C) decision
support.

> ⚠️ **Decision support, not regulatory advice.** ReguLumen is an engineering aid.
> Every output must be reviewed and confirmed by a qualified person. It does not
> produce a regulatory verdict.

---

## Problem & who it's for

Compliance engineers and QA/RA teams in medtech repeatedly ask questions like
*"What does the EU MDR require for medical-device software lifecycle and risk
management?"* or *"What IEC 62304 safety class applies to this component?"* —
and answer them by Ctrl-F-ing through 200-page PDFs. In a regulated context an
**unverifiable** answer is worse than no answer. ReguLumen answers in natural
language and shows the **exact clause** it relied on, or explicitly declines.

## What it does

- **Citation-faithful retrieval with strict abstention.** Every answer is
  grounded in retrieved clauses and cites them. A post-generation faithfulness
  gate rejects any answer that cites a clause it didn't actually retrieve (or a
  vague catch-all label) and turns it into an explicit abstention.
- **IEC 62304 risk-class decision support.** Given a software-component
  description, it suggests a safety class (A/B/C) using a transparent, auditable
  §4.3 decision table fed by LLM-extracted facts — and abstains when severity
  can't be established.
- **An evaluation harness.** A 30-item gold Q&A set plus retrieval and citation
  metrics, so retrieval quality is *measured*, not asserted.

## Architecture

A shared core with two thin entry points (the UI and API both import the same
`regulumen.core` in-process — the UI never depends on the API being up):

```text
        ┌──────────────┐        ┌──────────────┐
        │  Gradio UI   │        │   FastAPI    │   /ask  /classify  /health
        │ (in-process) │        │ router→svc→  │
        └──────┬───────┘        │  repository  │
               │                └──────┬───────┘
               └─────────┬─────────────┘
                         ▼
                  regulumen.core
   ingestion ─▶ retrieval ─▶ generation ─▶ risk
   (parse +     (Chroma dense  (LiteLLM/Groq  (IEC 62304 §4.3
   clause-aware  + BM25 lexical  + strict       table + LLM
   chunking)     hybrid fusion)  faithfulness)  fact extraction)
                         │
            HuggingFace Inference embeddings (hosted, key-light)
```

- **Ingestion** parses each source by format (EUR-Lex HTML; FDA/IMDRF PDF) and
  chunks by document structure, not token count, so a chunk maps to a precise
  clause and its citation never blurs two clause numbers.
- **Retrieval** fuses dense (ChromaDB) and lexical (BM25) results — pure vectors
  miss exact clause numbers like *"§5.1.1"*; BM25 catches them.
- **Generation** prompts for grounded-only, clause-cited answers, then a
  faithfulness gate independently verifies grounding. The Q&A flow is a
  **LangGraph state graph** (`retrieve → generate → gate → refine`) with a
  bounded refine loop: when the gate rejects an answer, the graph refines the
  query and re-retrieves once before abstaining — recovering answers whose
  correct clause existed in the index but was mis-ranked below top-k. The gate
  stays authoritative; the loop retries, never bypasses it. (Set
  `REGULUMEN_ANSWERER_MODE=linear` for the single-pass orchestrator.)
- **Risk** keeps the regulatorily-meaningful logic in a transparent pure-data
  table; the (non-deterministic) LLM only extracts facts, never picks a class.

## Design decisions

| Decision | Alternative considered | Why |
|---|---|---|
| Hybrid retrieval (dense + BM25) | Pure vector search | Vectors miss exact clause numbers ("62304 §5.1.1"); BM25 catches lexical matches. Both matter for regulatory lookups. |
| Clause-aware chunking | Fixed token-size chunking | Citations must be precise; a chunk is bounded by document structure so it maps to one clause and the citation never spans two clause numbers. |
| Strict abstention by default | Always answer with a confidence score | In a regulated context an unverifiable answer is a liability. Abstaining (and saying what's missing) is the safer, more defensible behavior. |
| LangGraph Q&A graph with a refine loop | Single linear retrieve→generate→gate pass | A state graph lets a failed faithfulness check trigger a bounded query-refine + re-retrieve instead of an immediate abstention — recovering mis-ranked clauses. Honest scope: it helps when the right clause is indexed but mis-ranked, not when the model cited a non-existent clause. Both orchestrators are config-selectable and share the same authoritative gate. |
| Hybrid rules + LLM for risk class | Pure-LLM classification | A subtly-wrong class is worse than abstaining. The §4.3 table is auditable pure data; the LLM only extracts facts, keeping the decision defensible and reproducible. |
| LiteLLM gateway | Direct provider SDK | Provider-agnostic: the same code runs against Groq, OpenAI, Anthropic, or local Ollama — a portability/cost lever. |
| Hosted HF Inference embeddings | Local sentence-transformers model | No multi-GB model download, leaner Docker image, no GPU — embeddings are a hosted HTTP call behind a swappable interface. |
| Shared core, two entry points | Gradio calls FastAPI over HTTP | The UI imports the core in-process, so the demo runs as a single process (e.g. on a HF Space) and the UI doesn't depend on the API being up. |

## Citation faithfulness

The differentiator. Two layers:

1. **Grounded-only prompting.** The model is instructed to answer *only* from the
   provided excerpts, cite the exact clause label in a machine-extractable form,
   use no outside knowledge, and emit an explicit abstention token when the
   excerpts don't contain the answer.
2. **A post-generation faithfulness gate** (independent of the model). It extracts
   the citations the answer used and verifies each one was actually retrieved. An
   answer is **grounded only if** it cites at least one clause, every cited label
   was retrieved, and no cited label is a vague catch-all (e.g. bare `[[EU MDR]]`
   matching everything). Otherwise the answer is replaced by a structured
   abstention naming what's missing.

**Honest limitation — citation-grounding ≠ claim-grounding.** The gate proves the
cited *label* was retrieved (preventing fabricated citations); it does **not**
verify that the surrounding prose is semantically entailed by that clause's text.
A model that cites a real clause but misstates its content would pass. Full
claim-grounding needs a second-pass NLI or LLM-judge step — see the roadmap.

## Evaluation

Built early so retrieval could be tuned against measured numbers rather than
guesswork.

- **Gold set:** `eval/gold/gold_qa.jsonl` — 30 items: 24 retrieval-answerable
  (across EU MDR Articles/Annexes and IMDRF definitions) + 6 must-abstain
  (out-of-corpus standards like IEC 62304, ISO 13485, ISO 14971). Schema and
  conventions in `eval/gold/schema.md`.
- **Metrics:** precision@k, recall@k, citation accuracy (boundary-aware so
  `§17.1` never falsely matches `§17.10`), and abstention correctness. Retrieval
  means are computed over the retrieval-scored items only, so must-abstain items
  don't inflate them.

**Results** below are from a real-embeddings run on 2026-06-03 (HuggingFace
Inference, `BAAI/bge-large-en-v1.5`, 1024-dim) over the 24 retrieval-scored gold
items (the 6 must-abstain items are excluded from retrieval means). Tuned
defaults: `hybrid_alpha=0.3`, `top_k=8` (see `docs/eval-tuning-notes.md`).
Reproduce with:

```bash
python scripts/ingest.py            # build the real index (needs REGULUMEN_HF_API_TOKEN)
python -m regulumen.core.eval.runner
```

| Metric | Result (k=8) |
|---|---|
| Recall@8 | 0.875 (21 / 24 expected clauses retrieved) |
| Citation accuracy | 0.875 |
| MRR | 0.78 |
| Precision@8 | 0.23 |

Precision@8 is low **by construction** — each question expects only 1–2 specific
clauses among 8 retrieved, so per-item precision caps around 0.12–0.25; recall
and citation accuracy are the meaningful retrieval signals. Recall is flat across
fusion weights: the 3 misses are clauses that don't surface in either retriever's
candidate pool (a chunking/query-handling gap, not a weighting one) — documented
honestly in the tuning notes rather than hidden by re-weighting.

## Local setup

ReguLumen uses two free-tier providers, configured via a `.env` file
(`REGULUMEN_`-prefixed):

- **Groq** (LLM) — get a key at <https://console.groq.com/keys>
- **HuggingFace Inference** (embeddings) — get a read token at
  <https://huggingface.co/settings/tokens>

Create `.env` in the project root:

```text
REGULUMEN_GROQ_API_KEY=gsk_...
REGULUMEN_HF_API_TOKEN=hf_...
```

### Option A — Docker (one-command spin-up)

```bash
docker compose --profile ingest run --rm ingest   # build the Chroma index
docker compose up                                 # start API + UI
```

- API → <http://localhost:8000> (`/ask`, `/classify`, `/health`, docs at `/docs`)
- UI → <http://localhost:7860>

The Chroma index persists in a `./.chroma` volume shared by the services.

### Option B — local Python (venv)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (use: source .venv/bin/activate  on Unix)
pip install -e .
pip install -r requirements.txt

python scripts/ingest.py          # build the real index (needs the HF token)
# or, with no token / no network — an offline smoke index:
python scripts/ingest.py --use-fake-embedder

uvicorn regulumen.api.main:app --port 8000   # API
python -m regulumen.ui.app                   # UI
```

A `Makefile` wraps the common tasks: `make install`, `make test`, `make lint`,
`make ingest`, `make ingest-fake`, `make api`, `make ui`, `make up`, `make down`.

### Refreshing the corpus

```bash
python scripts/fetch_corpus.py    # re-downloads the public sources + manifest
```

## Regulatory disclaimer & intellectual property

**ReguLumen provides decision support, not regulatory advice or a verdict.** All
outputs must be reviewed and confirmed by a qualified person.

The corpus uses only **publicly available** documents: the EU MDR (Regulation
(EU) 2017/745) Official Journal text, and FDA / IMDRF guidance documents.
Copyrighted standard texts — **IEC 62304, ISO 14971, ISO 13485** — are **not**
redistributed. For IEC 62304, only the §4.3 classification *decision logic* is
encoded (for decision support); the standard's text is not reproduced. Consult
your licensed copy of any standard for the authoritative requirements.

## Limitations & roadmap

**Known limitations (stated deliberately):**

- **FDA PDF parsing is weak for two sources.** EU MDR (HTML) and IMDRF (clean
  dotted sections) parse well. The FDA *Premarket Software Functions* PDF
  extracted as a webinar-transcript layout and degraded to a per-page fallback;
  the FDA *SaMD Clinical Evaluation* PDF lost its section numbering. Live testing
  showed the transcript's noisy chunks made the generator over-abstain even on
  good EU MDR questions, so degenerate parses are now dropped at ingest via
  `min_structure_confidence` (the FDA Premarket source is excluded; corpus is 869
  chunks). Sourcing cleaner FDA PDFs is on the roadmap.
- **Citation-grounding ≠ claim-grounding** (see above): the gate prevents
  fabricated citations but does not yet verify prose claims against clause text.

**Roadmap:**

- Real-embeddings evaluation numbers (run the harness with an HF token).
- Cleaner FDA source parsing (better source PDFs / OCR-aware extraction).
- Claim-grounding via an NLI / LLM-judge second pass.
- Deeper multi-step regulatory reasoning on the existing LangGraph graph (the
  Q&A refine loop is shipped; next is multi-hop reasoning across clauses).
- An MCP server to expose ReguLumen to other tools.
- LiteLLM Proxy for org-wide cost control.
- ~~Phase 2: a HuggingFace Space demo~~ — Docker Space deployment shipped
  (see [docs/HF_DEPLOY.md](docs/HF_DEPLOY.md)).

The full, prioritised task list lives in **[ROADMAP.md](ROADMAP.md)**.

## Project layout

```text
src/regulumen/
  core/
    ingestion/   parsers (EUR-Lex HTML, FDA/IMDRF PDF), clause-aware chunker, models
    retrieval/   embedder (HF Inference), Chroma vector store, BM25, hybrid fusion
    generation/  LiteLLM client, prompts, faithfulness gate, linear + LangGraph answerers
    risk/        IEC 62304 §4.3 severity table, hybrid classifier
    eval/        metrics, runner, gold-set models
    config.py    pydantic-settings (REGULUMEN_ env prefix)
  api/           FastAPI: schemas, repository, service, router, main
  ui/            Gradio app + pure markdown renderers
scripts/         fetch_corpus.py, ingest.py
eval/gold/       gold_qa.jsonl (30 items) + schema.md
corpus/          public sources + manifest.json (gitignored raw files)
tests/           mirrors src/ (run offline with fakes/mocks)
```

## Testing

```bash
python -m pytest -q     # 277 passed, 2 skipped
ruff check .            # clean
```

For the live end-to-end tests (real Groq + HuggingFace calls), set both keys and
run `python -m pytest -q -m live`.

The 2 skipped tests are live network checks (real Groq / HuggingFace calls),
gated on the API keys and skipped by default. Everything else runs **fully
offline** using deterministic fakes for the LLM and embeddings, so the suite is
fast and reproducible without any keys.

## License

[Apache License 2.0](LICENSE) © 2026 Unnikuttan Vijayan Velamkudy. Regulatory
source documents fetched by `scripts/fetch_corpus.py` remain under their
publishers' terms and are not redistributed in this repository.
