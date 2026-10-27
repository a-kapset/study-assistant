"""Builds the chat model of each role from configuration; the only module that knows about any model provider.

The rest of the app sees LangChain's provider-neutral chat model and the two error types below. A role names its
model as "provider/model", and this module picks the builder for that provider, so changing the model or the
provider is a change of configuration only.
"""

from collections.abc import Callable, Mapping

import httpx2
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from study_assistant.settings import Settings

# Repeats of a call that could not connect, timed out, was rate-limited or got a server error. Set here instead of
# left to the libraries, whose defaults differ and change between versions, so a provider that is down fails a
# learner's request within seconds.
_MAX_RETRIES = 2

type _Builder = Callable[[str, Settings, httpx2.AsyncClient], BaseChatModel]


class LlmError(Exception):
    """A call to a chat model failed; no answer was returned."""


class LlmUnavailableError(LlmError):
    """The model provider could not be reached, or did not answer in time, even after the repeats."""


class LlmRequestError(LlmError):
    """The model provider answered the call with an error status."""

    def __init__(self, status_code: int) -> None:
        """Keep the status the provider answered with; it is the only detail the message shows."""
        super().__init__(f"the model provider answered with status {status_code}")
        self.status_code = status_code


class UnknownProviderError(ValueError):
    """A role names a provider that the app has no builder for."""


def _build_openai(model: str, settings: Settings, http_client: httpx2.AsyncClient) -> BaseChatModel:
    """Build an OpenAI chat model that uses the Responses API and keeps nothing on the provider's side.

    The key and the base address are given explicitly, so OPENAI_API_KEY and OPENAI_API_BASE in the environment
    are never used. The library still adds OPENAI_ORG_ID if it is set; the stack passes no such variable.
    """
    openai_settings = settings.openai
    return ChatOpenAI(
        model_name=model,
        openai_api_key=openai_settings.api_key,
        openai_api_base=str(openai_settings.base_url),
        request_timeout=openai_settings.timeout_seconds,
        max_retries=_MAX_RETRIES,
        http_async_client=http_client,
        # Every call sends everything the model needs and nothing is stored by the provider, so a call can be
        # recorded and replayed on its own.
        use_responses_api=True,
        store=False,
    )


_BUILDERS: Mapping[str, _Builder] = {"openai": _build_openai}


def build_chat_model(model_ref: str, settings: Settings, http_client: httpx2.AsyncClient) -> BaseChatModel:
    """Build the chat model named by ``model_ref`` ("provider/model"); nothing is sent until the first call.

    A provider without a builder raises ``UnknownProviderError`` naming the providers that exist, so a typo in
    the configuration stops the app at startup instead of failing the first question.
    """
    provider, _, model = model_ref.partition("/")
    builder = _BUILDERS.get(provider)
    if builder is None:
        known = ", ".join(sorted(_BUILDERS))
        raise UnknownProviderError(f"unknown model provider {provider!r} in {model_ref!r}; known: {known}")
    return builder(model, settings, http_client)


async def ainvoke[Input, Output](runnable: Runnable[Input, Output], value: Input) -> Output:
    """Call a chat model, or a runnable built on one, and turn a provider's failure into an ``LlmError``.

    No connection or no answer in time raises ``LlmUnavailableError`` with the error's type; an error status from
    the provider raises ``LlmRequestError``. The provider's own exception stays attached as the cause.
    """
    try:
        return await runnable.ainvoke(value)
    except openai.APIConnectionError as error:
        # Also covers a timeout, which the library reports as a subclass of the connection error.
        raise LlmUnavailableError(type(error).__name__) from error
    except openai.APIStatusError as error:
        raise LlmRequestError(error.status_code) from error
