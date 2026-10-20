-- Embedding spaces: which model and prefixes made a set of vectors.
-- Each space gets its own vector table, created by the migration run from the configured model, because a vector
-- column has one fixed length and its index holds vectors of one space only. This file adds only what does not
-- depend on the configuration: the registry of spaces, and the key that vector tables point at.

-- One row per space that has a vector table. The id is the end of the table's name; the other fields are what the
-- id is made from, kept so a person can read which model and prefixes a table holds.
CREATE TABLE embedding_spaces (
    space_id       text PRIMARY KEY,
    model          text NOT NULL,
    revision       text NOT NULL,
    passage_prefix text NOT NULL,
    query_prefix   text NOT NULL,
    -- The length of every vector in the space's table.
    dimension      integer NOT NULL CHECK (dimension > 0),
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- A vector row repeats its chunk's course, so a search can filter by course in the table that holds the vector
-- index. The pair is a key of chunks, so a vector row can point at it and never name a course its chunk is not in.
ALTER TABLE chunks ADD UNIQUE (chunk_id, course_id);
