# Study Assistant

A study assistant that answers questions from course materials, runs practice exams and tracks
progress — together with an evaluation harness that tests and measures it from the outside.

> **Status: work in progress.** Both the app and the evaluation harness are under active development;
> the READMEs of each part describe what already works.

The repository holds two independent projects and the course definitions they work with:

| Part | What it is |
|---|---|
| [`app/`](app/README.md) | **Study Assistant** — a domain-independent web service: retrieval-augmented answers with citations, an exam engine, document uploads with OCR and LLM agents. It knows nothing about any particular subject: courses are data. |
| [`harness/`](harness/README.md) | **Evaluation harness** — tests and evaluates the app as a black box: only over HTTP, through configuration and through an LLM proxy. It never imports app code, so it can be pointed at other LLM applications with the same boundary. |
| [`courses/`](courses/README.md) | **Course definitions** — one manifest per course: metadata, structure rules and settings. Course materials themselves are never stored in the repository. |

Each project has its own environment, tooling and README with setup and usage details.

## License

[MIT](LICENSE)
