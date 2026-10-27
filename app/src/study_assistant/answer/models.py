"""What an answer to a learner's question holds: the text, whether it is a refusal to answer, and the course
passages it rests on."""

from pydantic import BaseModel, ConfigDict


class Citation(BaseModel):
    """A course passage an answer rests on, and where the learner can read it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_id: str
    # The section of the passage; None for text before the first heading of a file.
    section_id: str | None
    source_file: str
    page_start: int
    page_end: int


class Answer(BaseModel):
    """The answer the learner reads, with the passages it cites.

    Without a search (the ungrounded mode) an answer never declines and cites nothing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    answer: str
    # True when no course passage was close enough to the question, or the passages found do not answer it; the
    # answer then says so instead of answering.
    abstained: bool
    # The passages the answer cites, each once, in the order the model first cited them.
    citations: list[Citation]
    # Citations the model gave to passages it was not shown; they are left out of ``citations``.
    invalid_citations: int
