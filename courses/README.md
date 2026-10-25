# Courses

One folder per course, each with a `course.yaml` manifest: the course's metadata, the rules that
describe its structure, and the list of its source files. The manifests are committed; the course
materials themselves (syllabi, books, exams) never are and stay on the local machine.

> **Status: work in progress.** Both the app and the evaluation harness are under active development;
> the READMEs of each part describe what already works.

| Course | What it is |
|---|---|
| [`demo`](demo/course.yaml) | Synthetic course: the rules of an invented board game. Used by tests and local runs. |
| [`lumen_moths`](lumen_moths/course.yaml) | Synthetic course: an invented handbook with the shape of an exam syllabus and a bank of 102 questions. Used to measure search where the real courses are not available, such as CI. |
| [`istqb_ctfl`](istqb_ctfl/course.yaml) | ISTQB Certified Tester Foundation Level, syllabus v4.0.1, with a question bank from the four sample exam sets. |

## The manifest

| Field | Meaning |
|---|---|
| `id` | Lowercase letters, digits and `_`; must equal the folder name. |
| `title`, `description` | Shown to users; the description says what the course covers. |
| `language` | Two-letter code of the course material, e.g. `en`. |
| `structure` | Regular expressions that find `section_heading` and, optionally, `objective_id` and `knowledge_level` in the source text. A course without learning objectives (for example, a set of books) gives only `section_heading`. |
| `structure.section_heading` | Matched against a whole line. Named groups `id` and `title`, e.g. `'^Unit (?P<id>\d+)\. (?P<title>.+)$'`. When ids are numbers (`2`, `4.2.1`), only headings whose numbers keep increasing count, so a numbered list in the text does not open a section. |
| `structure.objective_id` | A line that starts with a match defines a learning objective. By default it belongs to the section it is printed in; a named group `section` says which section it belongs to instead, for a course that lists all objectives at the start of a chapter. |
| `structure.knowledge_level` | Searched in the objective's line; named group `level`. Needs `objective_id`. |
| `sources` | At least one entry: `file` (path relative to the course's source directory, see below), `role` (`material`, `glossary` or `question_bank`), `authority` (`primary` sources define what is correct; `supplementary` ones add background and never override them) and, optionally, `noise_lines`. |
| `sources[].content_sections` | Optional regular expression matched against a whole section id: only matching sections get chunks, so only their text can be found by search. The rest of the file (front matter, references, appendices, an index, a chapter's list of objectives) stays in the course structure but is not searched; text before the first heading has no id and is never searched. Without it every section is searched. |
| `sources[].pages` | Optional list of inclusive `[first, last]` page ranges, ascending and not overlapping: only these pages of the file are read, e.g. a textbook's teaching chapters without its practice exams and their answers. Page numbers stay those of the file, so citations name the real page. Without it every page is read. |
| `sources[].section_heading` | Optional heading pattern for this file only, with the same named groups as `structure.section_heading`, for a book whose headings are written differently from the course's main material. |
| `sources[].noise_lines` | Regular expressions for the lines a file's layout repeats on every page (running header, footer, page number). A line is dropped from the extracted text only when a pattern matches all of it, so a sentence that merely starts like a header stays. Each file has its own list, because each has its own layout. |

Unknown keys, invalid regular expressions and missing fields are errors: the app refuses to start and
names the course, the file and the problem.

## The demo course files

The `demo` and `lumen_moths` courses are the only ones whose material is committed, because their text is invented.

| File | Role |
|---|---|
| `demo/vardolin_rules.md` | The known text of the course. Tests compare the parsed and cleaned PDF with it word for word; it is not a course source itself. |
| `demo/vardolin_rules.pdf` | The course source the app reads, built from the Markdown with a running header and footer, words split at line ends and a figure-only page, like a real book. |
| `demo/tools/build_pdf.py` | Builds the PDF from the Markdown. The output is the same byte for byte on every run. |

The Markdown uses a small subset: `#` for the title, `##` for unit headings, paragraphs separated by
blank lines, `&shy;` where a word may be split at a line end, and `<!-- figure page -->` for a page
with a drawing and no text. Only ASCII text, because the PDF uses a built-in font.

## The lumen_moths course files

| File | Role |
|---|---|
| `lumen_moths/lumen_moths.pdf` | The handbook: 6 chapters, 30 sections, 12 of them split into subsections, 90 learning objectives with knowledge levels listed at the start of each chapter. |
| `lumen_moths/lumen_moths_bank.jsonl` | 102 multiple-choice questions in the common bank format; each names its learning objective, and so its answer section. |

The text and the questions were written once by a language model and are kept as they are; they are not
rebuilt. The questions paraphrase the text instead of copying it, so a search cannot find a passage just by
matching the question's wording.

After editing the Markdown, rebuild the PDF and commit both files. From the repository root
(Poetry runs the command inside `app/`):

```powershell
poetry -C app install --with tools
poetry -C app run python ../courses/demo/tools/build_pdf.py
poetry -C app run pytest -q
```

## Question banks

A course's exam questions are one more source in its manifest, with the role `question_bank`. Ingestion does not
cut a bank into chunks. Every course writes its bank in the same format: JSON Lines, one question per line.

| Field | Meaning |
|---|---|
| `id` | Unique in the bank; answers and results refer to the question by it. |
| `stem` | The question text, with any scenario or list it needs; line breaks kept. |
| `options` | Two or more answer options, each `{"id": "a", "text": "..."}`; the ids are unique. |
| `keys` | One or more option ids that are correct; a question with two keys asks to select both. |
| `objective_id` | The learning objective the question tests. |
| `knowledge_level` | Optional, e.g. `K2`. |
| `source` | Where a person finds the question: `document`, `version`, `question` (the number the document prints), optional `page`. |

The app reads a bank as a whole and stops at the first unusable line, naming the line and the reason:
`invalid_json` (the line is not JSON), `invalid_item` (a missing or unknown field, a key that is not one of the
options, repeated option ids or keys) or `duplicate_item_id`.

### The `istqb_ctfl` bank

The bank comes from the four official sample exam sets. Each set is two PDFs, the questions and the answers (a key
table with the correct options, the learning objective and the K-level of every question).
`istqb_ctfl/tools/convert_sample_exams.py` reads both and checks them against each other before it writes anything:
the same question numbers in both, as many keys as the question asks to select, every key among the options, the
options lettered a, b, c in order. On any difference it lists all of them and writes nothing; otherwise it writes the
bank, reads it back with the app's reader and prints the number of questions and the file's SHA-256. The PDFs and the
bank stay on this machine.

From the repository root (Poetry runs the command inside `app/`), with the folder that holds the sample exam PDFs and
the bank file in the course's source folder:

```powershell
poetry -C app run python ../courses/istqb_ctfl/tools/convert_sample_exams.py C:/materials/istqb/Sample_Exams C:/materials/istqb/sample_exams_bank.jsonl
```

The check catches a misread key table, but not a misread question text (for example, the second line of a long option
moved into the question). After a new conversion, compare a few questions with long options against the PDF. Figures
are not in the text: a question built on a diagram keeps only its words.

## Adding and enabling a course

1. Create `courses/<course_id>/course.yaml`; the app code does not change.
2. Enable it with `APP_COURSES__ENABLED` in the root `.env`, a comma-separated list of course ids, e.g.
   `APP_COURSES__ENABLED=demo,istqb_ctfl`; `.env.example` enables only `demo`.
3. Put its source files where the app finds them. The app reads `<source directory>/<course_id>/<file>`
   (`APP_COURSES__SOURCE_DIR`). In Docker, Compose mounts this folder (`courses/`, for the committed demo
   files) and one folder of a course whose material stays on this machine: set `LOCAL_COURSE_ID` to the
   course id and `LOCAL_COURSE_SOURCES` to the folder holding the files its manifest lists, e.g.
   `LOCAL_COURSE_SOURCES=C:/materials/istqb`. The files are not copied or renamed.
4. Ingest it: `POST /v1/admin/courses/<course_id>/ingest` with the admin key (see the app's README), then
   `GET /v1/courses/<course_id>` shows the stored counts.

## Why courses are data

The app knows nothing about a particular course. Everything course-specific (title,
description, structure rules, list of sources) lives in `courses/<course_id>/course.yaml`.

- **No course logic in the app.** App code and tests contain no course names; a pre-commit
  hook rejects them.
- **A new course is a new folder, not a code change.** Running the same code on a second
  course shows whether anything course-specific has leaked into it.
- **Tests use an invented course of a different shape.** The committed `demo` course is the
  rules of an invented board game, so a model cannot answer from what it already knows, and
  its structure is unlike ISTQB's, so tests exercise the generic code rather than code tuned
  to one course.
- **Course materials stay out of the repository.** Only the manifest is committed; source
  documents stay on the local machine. The invented `demo` and `lumen_moths` texts are the exceptions.
- **A broken manifest stops startup.** Manifests are validated when the app starts, and the
  error names the course and the file, instead of showing up later as wrong answers.

## How the app uses a manifest

The app code knows the general course model, but no particular course:

| The app code knows | Each manifest says |
|---|---|
| What a course, source, section, learning objective, knowledge level and question bank are | The course's id, title, description and language |
| How to read a PDF, clean its text, split it into chunks, search them and run an exam | Which files belong to the course, and each file's role and authority |
| | What section headings and, if the course has them, objective ids and knowledge levels look like in its text (regular expressions) |
| | Which repeated lines (header, footer, page number) to drop from each file |

For example, with the `demo` course:

1. The app reads `demo/course.yaml` and finds one primary source, `vardolin_rules.pdf`.
2. The parser, the same for every course, extracts the text of each page.
3. Text cleaning drops lines such as `Page 3 of 7`, because the manifest lists that pattern for
   this file.
4. The structure rules find `Unit 2. Taking a turn` as section `2` and `OBJ-02-03` as an objective
   of that section.
5. The sections are cut into chunks, and ingestion stores the document, its sections, objectives and chunks
   under the course id. Editing any of the manifest values used in steps 1 to 4 rebuilds the document on the
   next ingestion; a new title or description does not.

The ISTQB course goes through the same code; only its manifest differs.

Courses are set up by the operator of an installation. Learners do not write manifests.
