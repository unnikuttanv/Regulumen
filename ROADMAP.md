# Roadmap

Planned work, roughly in priority order. Effort estimates are rough.

## Now — answer quality

- [ ] **Cleaner FDA source parsing** (~6 h): replace the two badly-parsed FDA PDFs (Premarket Software Functions, SaMD Clinical Evaluation) with cleaner sources or OCR-aware, layout-aware extraction, so section numbering survives and the Premarket source can be re-admitted to the corpus.
- [ ] **Claim-grounding pass** (~6 h): extend the faithfulness gate so it checks that each sentence of the answer is supported by the text of the clause it cites, not just that the citation was retrieved; abstain or trim unsupported sentences.
- [ ] **Multi-hop regulatory reasoning** (~6 h): extend the existing LangGraph refine loop to follow cross-references between clauses (e.g. MDR Annex I → Article → harmonised standard reference).
- [ ] **Real-embeddings regression run in CI** (~3 h): scheduled GitHub Actions job (with an HF token secret) that rebuilds the index and re-runs the retrieval test set, failing on regressions.

## Next — integration

- [ ] **MCP server** (~5 h): expose search, Q&A-with-citations and IEC 62304 classification as MCP tools so ReguLumen can be used from other assistants and IDEs.
- [ ] **LiteLLM Proxy** (~3 h): optional proxy config for shared cost control and model routing.
- [ ] **Corpus expansion** (~5 h): add further public sources (e.g. MDCG guidance documents) through the existing source registry, with parse-quality checks.

## Later — usability

- [ ] **Export** (~3 h): download an answer with its cited clauses as Markdown/PDF for review records.
- [ ] **Conversation history** in the Gradio UI.
- [ ] **Space hardening**: health check, cold-start index caching, usage limits on the public demo.
- [ ] Docs: architecture diagram and a contributor guide (adding a new regulatory source end to end).
