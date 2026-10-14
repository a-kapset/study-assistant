-- The course manifest's rules also shape a document's rows: the language, the structure patterns, the source's
-- role, authority and noise lines. Their hash is kept with the document, so editing them makes the next
-- ingestion rebuild the document instead of keeping rows made under the old rules.
-- No document is stored before ingestion exists, so the column needs no default for older rows.
ALTER TABLE documents ADD COLUMN rules_sha256 text NOT NULL;
