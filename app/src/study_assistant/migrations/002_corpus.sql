-- The course corpus: what ingestion stores for each course and how its rows depend on each other.
-- Every row carries its course id, so one course can be re-ingested or removed without touching another.
-- A document's sections, objectives and chunks are deleted with it, so replacing a changed source file
-- cannot leave rows of its old edition behind.

-- One row per ingested course; title and language are copied from its manifest at ingestion.
CREATE TABLE courses (
    course_id   text PRIMARY KEY,
    title       text NOT NULL,
    language    text NOT NULL
);

-- One row per source file of a course, keyed by its path in the manifest.
CREATE TABLE documents (
    course_id       text NOT NULL REFERENCES courses ON DELETE CASCADE,
    source_file     text NOT NULL,
    -- The edition of the file and the rules that produced its rows: both equal means re-ingesting changes nothing.
    sha256          text NOT NULL,
    chunker_version text NOT NULL,
    role            text NOT NULL,
    authority       text NOT NULL,
    parser          text NOT NULL,
    parser_version  text NOT NULL,
    engine_version  text NOT NULL,
    page_count      integer NOT NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (course_id, source_file)
);

-- Sections in reading order; the text before the first heading is a section without id.
CREATE TABLE sections (
    course_id   text NOT NULL,
    source_file text NOT NULL,
    position    integer NOT NULL,
    section_id  text,
    parent_id   text,
    title       text NOT NULL,
    page_start  integer NOT NULL,
    page_end    integer NOT NULL,
    PRIMARY KEY (course_id, source_file, position),
    -- Chunks and objectives point at a section by its id, so an id must name one section of the file.
    UNIQUE (course_id, source_file, section_id),
    FOREIGN KEY (course_id, source_file) REFERENCES documents ON DELETE CASCADE
);

-- Learning objectives, with the line that defines each one.
CREATE TABLE learning_objectives (
    course_id       text NOT NULL,
    source_file     text NOT NULL,
    objective_id    text NOT NULL,
    section_id      text,
    knowledge_level text,
    page            integer NOT NULL,
    line            text NOT NULL,
    PRIMARY KEY (course_id, source_file, objective_id),
    FOREIGN KEY (course_id, source_file) REFERENCES documents ON DELETE CASCADE,
    FOREIGN KEY (course_id, source_file, section_id) REFERENCES sections (course_id, source_file, section_id)
);

-- Chunks in reading order within their file. The id is derived from the content, so the same file and rules
-- give the same ids on every ingestion. The file's hash, authority and chunking rules are read from its document.
CREATE TABLE chunks (
    chunk_id      text PRIMARY KEY,
    course_id     text NOT NULL,
    source_file   text NOT NULL,
    position      integer NOT NULL,
    section_id    text,
    -- The objectives of the chunk's section and of the sections that enclose it.
    objective_ids text[] NOT NULL,
    page_start    integer NOT NULL,
    page_end      integer NOT NULL,
    language      text NOT NULL,
    text          text NOT NULL,
    -- Also the index that finds a document's chunks, to read them in order and to delete them with it.
    UNIQUE (course_id, source_file, position),
    FOREIGN KEY (course_id, source_file) REFERENCES documents ON DELETE CASCADE,
    FOREIGN KEY (course_id, source_file, section_id) REFERENCES sections (course_id, source_file, section_id)
);
