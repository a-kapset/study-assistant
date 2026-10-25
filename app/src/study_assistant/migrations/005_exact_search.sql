-- Course search is exact: a query is compared with every vector of the course. With the approximate HNSW index
-- the migration run used to create for each embedding space, PostgreSQL could switch to approximate search
-- whenever its table statistics changed, for example after an ingestion, and that search can miss a true nearest
-- passage. Exact search is fast enough for course-sized corpora, so the indexes are dropped; new spaces get none.
DO $$
DECLARE
    index_name text;
BEGIN
    FOR index_name IN
        SELECT indexname FROM pg_indexes
        WHERE schemaname = current_schema() AND indexname LIKE 'vectors\_%\_hnsw'
    LOOP
        EXECUTE format('DROP INDEX %I', index_name);
    END LOOP;
END
$$;
