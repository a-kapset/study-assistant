-- Search by words: every chunk keeps its text as a full-text search vector, the stems of its words with their
-- positions, so a search matches and ranks stems instead of parsing every text again. No index is built: a search
-- by words looks only at the chunks that already reached the lowest similarity to the question, a few hundred at
-- most, and an index on all chunks would not be read.

-- The text search configuration for a language code of a course manifest. A configuration knows the language's
-- stop words and how to cut its words to stems; "simple" only lower-cases words, so a course in a language without
-- an entry here is still searched by words, just without stemming. The function is immutable so the generated
-- column and the queries can use it, and both use the same configuration for the same language.
CREATE FUNCTION text_search_config(language text) RETURNS regconfig
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    RETURN CASE language WHEN 'en' THEN 'english'::regconfig ELSE 'simple'::regconfig END;

-- Computed by PostgreSQL from the chunk's text and language whenever a chunk is written, so it can never be out of
-- date with the text; adding it computes it for the chunks already stored.
ALTER TABLE chunks
    ADD COLUMN text_search tsvector GENERATED ALWAYS AS (to_tsvector(text_search_config(language), text)) STORED;
