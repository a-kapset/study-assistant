# Courses

One folder per course, each with a `course.yaml` manifest: the course's metadata, the rules that
describe its structure, and the list of its source files. The manifests are committed; the course
materials themselves (syllabi, books, exams) never are and stay on the local machine.

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
| `structure` | Regular expressions that find `section_heading`, `objective_id` and, optionally, `knowledge_level` in the source text. |
| `sources` | At least one entry: `file` (path relative to the course's source directory), `role` (`material`, `glossary` or `question_bank`) and `authority` (`primary` sources define what is correct; `supplementary` ones add background and never override them). |

Unknown keys, invalid regular expressions and missing fields are errors: the app refuses to start and
names the course, the file and the problem.

## Adding and enabling a course

1. Create `courses/<course_id>/course.yaml`; the app code does not change.
2. Enable it with `APP_COURSES__ENABLED`, a comma-separated list of course ids, e.g.
   `APP_COURSES__ENABLED=demo,istqb_ctfl`. No course is enabled by default; Docker Compose enables
   `demo` unless the root `.env` says otherwise.

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
  documents stay on the local machine.
- **A broken manifest stops startup.** Manifests are validated when the app starts, and the
  error names the course and the file, instead of showing up later as wrong answers.
