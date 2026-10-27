"""Answers to a learner's question about a course, written by the answer model.

For now the model answers from what it knows, without course passages. The same questions answered this way show
how much a model knows of a course without search, so answers built on search can be compared with them.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from study_assistant.courses.models import CourseManifest
from study_assistant.llm.providers import LlmError, ainvoke

# The course is named by its manifest, so the same instructions serve every course.
_TUTOR_PROMPT = (
    'You are a tutor for the course "{title}". The course covers: {description}\n'
    "Answer the learner's question about this course clearly and briefly."
)


class EmptyAnswerError(LlmError):
    """The model replied, but its reply has no text, for example because the model refused to answer."""

    def __init__(self) -> None:
        """Say what was wrong with the reply; the reply itself is not shown."""
        super().__init__("the model replied without any answer text")


class Answerer:
    """Ask the answer model a learner's question about a course and return the text of its answer."""

    def __init__(self, model: BaseChatModel) -> None:
        """Keep the chat model that writes the answers."""
        self._model = model

    async def answer(self, course: CourseManifest, question: str) -> str:
        """Return the model's answer to ``question`` about ``course``.

        The question goes to the model as the learner's own message and never becomes part of the instructions.
        A failed call raises an ``LlmError`` from the provider module; a reply without text raises
        ``EmptyAnswerError``, so a refusal never looks like an answer.
        """
        messages: list[BaseMessage] = [
            SystemMessage(_TUTOR_PROMPT.format(title=course.title, description=course.description)),
            HumanMessage(question),
        ]
        reply = await ainvoke(self._model, messages)
        text = reply.text.strip()
        if not text:
            raise EmptyAnswerError
        return text
