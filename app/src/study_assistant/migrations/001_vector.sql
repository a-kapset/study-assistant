-- The pgvector extension: the vector column type and the nearest-neighbour index methods.
-- Only a superuser can install it, so it is created here, by the migration run, not by the app server.
CREATE EXTENSION IF NOT EXISTS vector;
