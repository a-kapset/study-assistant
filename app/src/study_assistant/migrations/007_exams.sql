-- Learners, the question bank of each course, practice sessions and the answers given in them.

-- One row per learner. The app never stores a learner's key, only its SHA-256, so the table alone lets nobody in.
CREATE TABLE learners (
    learner_id  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    key_sha256  text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- The questions of a course's bank, in the order of its files. A question that left the bank is kept and marked
-- retired: sessions that already hold it stay readable, new sessions do not get it.
CREATE TABLE bank_items (
    course_id    text NOT NULL REFERENCES courses ON DELETE CASCADE,
    item_id      text NOT NULL,
    position     integer NOT NULL,
    stem         text NOT NULL,
    -- The answer options in their printed order: [{"id": "a", "text": "..."}, ...].
    options      jsonb NOT NULL,
    keys         text[] NOT NULL,
    objective_id text NOT NULL,
    retired      boolean NOT NULL DEFAULT false,
    PRIMARY KEY (course_id, item_id)
);

-- One practice session of one learner in one course.
CREATE TABLE exam_sessions (
    session_id  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    learner_id  uuid NOT NULL REFERENCES learners ON DELETE CASCADE,
    course_id   text NOT NULL REFERENCES courses ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX exam_sessions_learner_course ON exam_sessions (learner_id, course_id);

-- The questions of a session, in the order they are asked; a question appears once in a session.
CREATE TABLE exam_session_items (
    session_id  uuid NOT NULL REFERENCES exam_sessions ON DELETE CASCADE,
    position    integer NOT NULL,
    course_id   text NOT NULL,
    item_id     text NOT NULL,
    PRIMARY KEY (session_id, position),
    UNIQUE (session_id, item_id),
    FOREIGN KEY (course_id, item_id) REFERENCES bank_items
);

-- The answer log: at most one answer per question of a session, so sending an answer again cannot count twice.
-- Whether it was correct and which objective it tested are fixed when the answer is given, so a learner's
-- progress is a sum over these rows and does not change when the bank is edited later.
CREATE TABLE exam_answers (
    session_id   uuid NOT NULL,
    item_id      text NOT NULL,
    selected     text[] NOT NULL,
    correct      boolean NOT NULL,
    objective_id text NOT NULL,
    answered_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, item_id),
    FOREIGN KEY (session_id, item_id) REFERENCES exam_session_items (session_id, item_id) ON DELETE CASCADE
);
