# Courses

One folder per course, each with a `course.yaml` manifest: the course's metadata, the rules that
describe its structure, and the list of its source files. The manifests are committed; the course
materials themselves (syllabi, books, exams) never are and stay on the local machine.

> **Status: work in progress.** Both the app and the evaluation harness are under active development;
> the READMEs of each part describe what already works.

| Course | What it is |
|---|---|
| [`demo`](demo/course.yaml) | Synthetic course: the rules of an invented board game. Used by tests and local runs. |
| [`istqb_ctfl`](istqb_ctfl/course.yaml) | ISTQB Certified Tester Foundation Level, syllabus v4.0.1. |

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
| `sources[].noise_lines` | Regular expressions for the lines a file's layout repeats on every page (running header, footer, page number). A line is dropped from the extracted text only when a pattern matches all of it, so a sentence that merely starts like a header stays. Each file has its own list, because each has its own layout. |

Unknown keys, invalid regular expressions and missing fields are errors: the app refuses to start and
names the course, the file and the problem.

## The demo course files

The `demo` course is the only one whose material is committed, because its text is invented.

| File | Role |
|---|---|
| `demo/vardolin_rules.md` | The known text of the course. Tests compare the parsed and cleaned PDF with it word for word; it is not a course source itself. |
| `demo/vardolin_rules.pdf` | The course source the app reads, built from the Markdown with a running header and footer, words split at line ends and a figure-only page, like a real book. |
| `demo/tools/build_pdf.py` | Builds the PDF from the Markdown. The output is the same byte for byte on every run. |

The Markdown uses a small subset: `#` for the title, `##` for unit headings, paragraphs separated by
blank lines, `&shy;` where a word may be split at a line end, and `<!-- figure page -->` for a page
with a drawing and no text. Only ASCII text, because the PDF uses a built-in font.

After editing the Markdown, rebuild the PDF and commit both files. From the repository root
(Poetry runs the command inside `app/`):

```powershell
poetry -C app install --with tools
poetry -C app run python ../courses/demo/tools/build_pdf.py
poetry -C app run pytest -q
```

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
  documents stay on the local machine. The invented `demo` text is the one exception.
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
