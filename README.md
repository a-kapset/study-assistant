# Study Assistant

A study assistant that answers questions from course materials, runs practice exams and tracks
progress — together with an evaluation harness that tests and measures it from the outside.

> **Status: work in progress.** Both the app and the evaluation harness are under active development;
> the READMEs of each part describe what already works.

The repository holds two independent projects and the course definitions they work with:

| Part | What it is |
|---|---|
| [`app/`](app/README.md) | **Study Assistant** — a domain-independent web service: retrieval-augmented answers with citations, an exam engine, document uploads with OCR and LLM agents. It knows nothing about any particular subject: courses are data. |
| [`evaluation-harness/`](evaluation-harness/README.md) | **Evaluation harness** — tests and evaluates the app from the outside: over HTTP, through configuration, an LLM proxy and the app's data stores. It never imports app code. |
| [`courses/`](courses/README.md) | **Course definitions** — one manifest per course: metadata, structure rules and settings. Course materials themselves are never stored in the repository. |

Each project has its own environment, tooling and README with setup and usage details.

How the parts are tested and evaluated, and why each check sits where it does: [TESTING.md](TESTING.md).

## Plan

What is planned and what is done, in the order of work.

**1. Foundations**
- [x] App: service skeleton — health, readiness, effective configuration, request ids, error envelope, JSON logs
- [x] App: PostgreSQL with pgvector, migrations, least-privilege database roles
- [x] Evaluation harness: separate project; an HTTP transport that records every outcome, including failures; API tests
- [x] CI: lint, type checks, unit, integration and API tests on every push

**2. Courses as data and ingestion**
- [x] App: course manifests (metadata, structure rules, settings); course materials stay outside the repository
- [x] App: PDF parsing, structure-aware chunking, embeddings from a local embedding server, one vector table per embedding space
- [x] App: question bank in a common format
- [x] App tests: property-based chunking tests; integration tests on a real PostgreSQL
- [x] Evaluation harness: checks of the running stack's embedding setup

**3. Search**
- [x] App: course-scoped vector search with a typed result
- [x] Evaluation harness: retrieval gold sets; recall@k and MRR; misses told apart from irrelevant results
- [x] Evaluation harness: corpus changes measured before and after on the same questions
- [x] Evaluation harness: a synthetic course with a recall gate in CI

**4. Grounded answers**
- [x] App: provider-neutral model factory; answers grounded in the retrieved passages, with citations; a decline without a model call when nothing relevant is found
- [x] Evaluation harness: LLM proxy — record and replay of model calls, cost ledger
- [x] Evaluation harness: question-and-answer set; answer oracles (declines, valid citations, the right section cited); a prompt-injection case

**5. Statistics of evaluation results**
- [x] Evaluation harness: confidence intervals; target verdicts on the lower bound
- [x] Evaluation harness: paired comparison of two runs with a verdict (better, worse, no meaningful change, underpowered)
- [x] Evaluation harness: repeated answer runs and the flip rate
- [x] Evaluation harness: metamorphic relation — questions reworded without the course term

**6. Reliable evaluation runs**
- [ ] Evaluation harness: app version and settings in every run record
- [ ] Evaluation harness: fault injection in the LLM proxy; failure classifier (infrastructure or model); spending limit

**7. Tracing**
- [ ] App: OpenTelemetry traces per request and per stage

**8. Hybrid search and reranker**
- [ ] App: lexical search, hybrid fusion, cross-encoder reranker
- [ ] Evaluation harness: variants compared with paired statistics; retrieval relations (threshold, k, insertion order); latency per stage; the stage where a relevant passage was lost

**9. Mutant experiment**
- [ ] Evaluation harness: a catalogue of configuration and boundary faults; which check catches which fault; a report reproducible from run ids

**10. Practice exams and progress**
- [ ] App: practice sessions, scoring by course rules, progress per learning objective
- [ ] Evaluation harness: exact-answer oracle; API contract snapshot; stateful property tests of sessions

**11. Uploads and OCR**
- [ ] App: OCR service (Tesseract), learner uploads, failed pages reported
- [ ] Evaluation harness: OCR metamorphic suite (text PDF and its scan, rescaling); how OCR errors reach the answers

**12. Agents**
- [ ] App: tutor agent with a course-search tool; glossary tool; examiner agent with read-only text-to-SQL; router with an entry guard; conversation memory; agent-level injection guard; supervisor variant
- [ ] Evaluation harness: trajectory oracles; tool-selection matrix; tool-misuse suite; routing and guard confusion matrices; multi-turn cases; injection suite; router versus supervisor comparison

**13. Answer quality with an LLM judge**
- [ ] App: explanations of wrong answers; generated practice questions; learner memory built from progress data
- [ ] Evaluation harness: LLM judge with calibration; DeepEval metrics checked against own oracles and mutants; validity and leakage of generated questions; memory consistency

**14. CI pipeline, UI and monitoring**
- [ ] Evaluation harness: layered evaluation pipeline in CI with quality gates
- [ ] App and evaluation harness: minimal web UI with Playwright end-to-end tests
- [ ] Evaluation harness: Langfuse, online scoring, alerts, drift checks, load tests
- [ ] Evaluation harness: quality and cost report ending in a ship or hold decision

## License

[MIT](LICENSE)
