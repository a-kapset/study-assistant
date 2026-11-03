"""What an answer to a learner's question holds: the text, why it declines when it does, and the course passages it
rests on."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# Why an answer declines: no course passage was close enough, so the model was not called ("no_passages"), or the
# model read the passages found and said they do not answer the question ("model_declined").
AbstainReason = Literal["no_passages", "model_declined"]


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
    # Why the answer declines instead of answering; None when it answers. The answer text then says so.
    abstain_reason: AbstainReason | None
    # The passages the answer cites, each once, in the order the model first cited them.
    citations: list[Citation]
    # Citations the model gave to passages it was not shown; they are left out of ``citations``.
    invalid_citations: int
