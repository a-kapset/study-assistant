"""Explanations of wrongly answered practice questions, written by the answer model from course passages.

The question, its key and the learner's stored selection are facts of the question bank and the session; the model
only adds the reasons, and only from the passages a search of the course finds for the question. When no passage is
close enough the explanation is declined without calling the model, as an answer to a learner's question is.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from opentelemetry.trace import Tracer
from pydantic import BaseModel, ConfigDict, ValidationError

from study_assistant.answer.models import AbstainReason
from study_assistant.answer.prompts import passages_block
from study_assistant.answer.service import EmptyAnswerError, MalformedAnswerError, valid_citations
from study_assistant.courses.models import CourseManifest
from study_assistant.exams.service import StoredItem
from study_assistant.explain.prompts import explanation_instructions, question_block, search_query
from study_assistant.llm.providers import ainvoke
from study_assistant.retrieval.models import SearchHit
from study_assistant.retrieval.service import CourseSearch
from study_assistant.settings import SearchSettings
from study_assistant.tracing import stage_span

# The span of the request's trace that covers the model call and the reading of its reply.
GENERATE_SPAN = "generate_explanation"

# What the learner reads when no course passage is close enough to the question.
NO_PASSAGES_EXPLANATION = "The course material has no passage close enough to this question, so it is not explained."
# What the learner reads when the search could not compare the question with the course by meaning.
SEARCH_DEGRADED_EXPLANATION = (
    "The course search is running in a reduced mode right now, so this question is not explained. "
    "Please try again later."
)


class ShownPassage(BaseModel):
    """A course passage the model was shown for an explanation, and where the learner can read it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_id: str
    # The section of the passage; None for text before the first heading of a file.
    section_id: str | None
    source_file: str
    page_start: int
    page_end: int
    text: str
    # Whether the model named this passage as one its explanation uses.
    cited: bool


class Explanation(BaseModel):
    """The explanation the learner reads, with everything the model was shown to write it."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    explanation: str
    # Why the explanation declines instead of explaining; None when it explains. The text then says so.
    abstain_reason: AbstainReason | None
    # Every passage the model was shown, in the order shown; empty when the model was not called.
    passages: list[ShownPassage]
    # Citations the model gave to passages it was not shown; they mark no passage as cited.
    invalid_citations: int


class _ExplanationReply(BaseModel):
    """The reply the model is asked for; the instructions name these fields."""

    model_config = ConfigDict(extra="forbid")
    explanation: str
    # Numbers of the passages the explanation uses, as given in the passages block.
    citations: list[int]
    # True when the model says the passages do not explain the key; the explanation then says so.
    abstained: bool


class Explainer:
    """Explain a wrongly answered practice question from the passages of its course."""

    def __init__(
        self, model: BaseChatModel, search: CourseSearch, search_settings: SearchSettings, tracer: Tracer
    ) -> None:
        """Keep the model, the search and the tracer; the search runs with the default number of hits and lowest
        score. Nothing is sent until the first explanation."""
        self._tracer = tracer
        self._model = model.with_structured_output(_ExplanationReply, method="json_schema", strict=True)
        self._search = search
        self._k = search_settings.default_k
        self._min_score = search_settings.min_score

    async def explain(self, course: CourseManifest, item: StoredItem, selected: frozenset[str]) -> Explanation:
        """Return the explanation of ``item`` for a learner who selected ``selected`` instead of its key.

        The course is searched with the question's stem and the texts of its key. A search ranked by words only,
        or one that finds nothing close enough, is declined without calling the model: with no passage checked
        against the lowest score the model would explain from what it knows. The instructions, the passages and
        the question go to the model as three messages, so course text never becomes part of the instructions.
        A reply that does not have the asked-for form raises ``MalformedAnswerError``, one without text
        ``EmptyAnswerError``; a failed model call raises an ``LlmError`` and a failed search the search's own
        errors. The model call and the reading of its reply are one span of the request's trace.
        """
        result = await self._search.search(course.id, search_query(item), self._k, self._min_score)
        if result.degraded_reason == "embedder_unavailable":
            return _declined(SEARCH_DEGRADED_EXPLANATION, "search_degraded")
        if not result.hits:
            return _declined(NO_PASSAGES_EXPLANATION, "no_passages")
        messages: list[BaseMessage] = [
            SystemMessage(explanation_instructions(course)),
            HumanMessage(passages_block(result.hits)),
            HumanMessage(question_block(item, selected)),
        ]
        with stage_span(self._tracer, GENERATE_SPAN):
            try:
                reply = await ainvoke(self._model, messages)
            except ValidationError as error:
                raise MalformedAnswerError from error
            if not isinstance(reply, _ExplanationReply):
                raise MalformedAnswerError
            text = reply.explanation.strip()
            if not text:
                raise EmptyAnswerError
            cited, invalid = valid_citations(reply.citations, len(result.hits))
        reason: AbstainReason | None = "model_declined" if reply.abstained else None
        passages = [_shown(hit, number in cited) for number, hit in enumerate(result.hits, start=1)]
        return Explanation(explanation=text, abstain_reason=reason, passages=passages, invalid_citations=invalid)


def _declined(text: str, reason: AbstainReason) -> Explanation:
    """Return an explanation that declines with ``text`` for ``reason``, written without the model."""
    return Explanation(explanation=text, abstain_reason=reason, passages=[], invalid_citations=0)


def _shown(hit: SearchHit, cited: bool) -> ShownPassage:
    """Return the passage of ``hit`` as the reply carries it, marked as cited or not."""
    return ShownPassage(
        chunk_id=hit.chunk_id,
        section_id=hit.section_id,
        source_file=hit.source_file,
        page_start=hit.page_start,
        page_end=hit.page_end,
        text=hit.text,
        cited=cited,
    )
