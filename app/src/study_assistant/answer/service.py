"""Answers to a learner's question about a course, written by the answer model.

In the grounded mode the model answers only from the course passages a search finds and cites them by number; when
no passage is close enough, the question is declined without calling the model. In the ungrounded mode the model
answers from what it knows, without a search: the same questions answered both ways show how much the passages add.
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from study_assistant.answer.models import Answer, Citation
from study_assistant.answer.prompts import grounded_instructions, passages_block, tutor_instructions
from study_assistant.courses.models import CourseManifest
from study_assistant.llm.providers import LlmError, ainvoke
from study_assistant.retrieval.models import SearchHit
from study_assistant.retrieval.service import VectorSearch
from study_assistant.settings import AskSettings, SearchSettings

# What the learner reads when no course passage is close enough to the question.
NO_PASSAGES_ANSWER = "The course material has no passage close enough to this question, so it is not answered."


class EmptyAnswerError(LlmError):
    """The model replied, but its reply has no text, for example because the model refused to answer."""

    def __init__(self) -> None:
        """Say what was wrong with the reply; the reply itself is not shown."""
        super().__init__("the model replied without any answer text")


class MalformedAnswerError(LlmError):
    """The model's reply does not have the form it was asked for."""

    def __init__(self) -> None:
        """Say what was wrong with the reply; the reply itself is not shown."""
        super().__init__("the model's reply does not have the form it was asked for")


class _GroundedReply(BaseModel):
    """The reply the model is asked for in the grounded mode; the instructions name these fields."""

    model_config = ConfigDict(extra="forbid")
    answer: str
    # Numbers of the passages the answer uses, as given in the passages block.
    citations: list[int]
    abstained: bool


class Answerer:
    """Answer a learner's question about a course in the configured mode."""

    def __init__(
        self, model: BaseChatModel, search: VectorSearch, ask: AskSettings, search_settings: SearchSettings
    ) -> None:
        """Keep the model, the search and the mode; the grounded mode searches with the default number of hits and
        lowest score. Nothing is sent until the first question."""
        self._model = model
        self._grounded_model = model.with_structured_output(_GroundedReply, method="json_schema", strict=True)
        self._search = search
        self._mode = ask.mode
        self._k = search_settings.default_k
        self._min_score = search_settings.min_score

    async def answer(self, course: CourseManifest, question: str) -> Answer:
        """Return the answer to ``question`` about ``course``.

        A failed model call raises an ``LlmError``; a failed search raises the search's own errors.
        """
        if self._mode == "ungrounded":
            text = await self._answer_from_memory(course, question)
            return Answer(answer=text, abstained=False, citations=[], invalid_citations=0)
        return await self._answer_from_passages(course, question)

    async def _answer_from_memory(self, course: CourseManifest, question: str) -> str:
        """Ask the model without any course passage and return the text of its reply.

        A reply without text raises ``EmptyAnswerError``, so a refusal never looks like an answer.
        """
        messages: list[BaseMessage] = [SystemMessage(tutor_instructions(course)), HumanMessage(question)]
        reply = await ainvoke(self._model, messages)
        text = reply.text.strip()
        if not text:
            raise EmptyAnswerError
        return text

    async def _answer_from_passages(self, course: CourseManifest, question: str) -> Answer:
        """Search the course, and answer only from the passages found, or decline when none is close enough.

        The passages go to the model in their own message, after the instructions and before the question, so
        neither the course text nor the learner's question becomes part of the instructions. A reply that does not
        have the asked-for form raises ``MalformedAnswerError``; one without answer text ``EmptyAnswerError``.
        """
        result = await self._search.search(course.id, question, self._k, self._min_score)
        if result.status != "ok":
            # Nothing close enough was found: the model is not called, so it cannot answer from what it knows.
            return Answer(answer=NO_PASSAGES_ANSWER, abstained=True, citations=[], invalid_citations=0)
        messages: list[BaseMessage] = [
            SystemMessage(grounded_instructions(course)),
            HumanMessage(passages_block(result.hits)),
            HumanMessage(question),
        ]
        try:
            reply = await ainvoke(self._grounded_model, messages)
        except ValidationError as error:
            raise MalformedAnswerError from error
        if not isinstance(reply, _GroundedReply):
            raise MalformedAnswerError
        text = reply.answer.strip()
        if not text:
            raise EmptyAnswerError
        citations, invalid = _cite(reply.citations, result.hits)
        return Answer(answer=text, abstained=reply.abstained, citations=citations, invalid_citations=invalid)


def _cite(numbers: Sequence[int], hits: Sequence[SearchHit]) -> tuple[list[Citation], int]:
    """Turn the passage numbers the model cited into citations and count the numbers that name no passage.

    Each passage is cited once, in the order the model first cited it; every number outside 1..len(hits) is left
    out and counted, repeats included.
    """
    cited: list[int] = []
    invalid = 0
    for number in numbers:
        if not 1 <= number <= len(hits):
            invalid += 1
        elif number not in cited:
            cited.append(number)
    return [_citation(hits[number - 1]) for number in cited], invalid


def _citation(hit: SearchHit) -> Citation:
    """Return where the learner can read the passage of ``hit``."""
    return Citation(
        chunk_id=hit.chunk_id,
        section_id=hit.section_id,
        source_file=hit.source_file,
        page_start=hit.page_start,
        page_end=hit.page_end,
    )
