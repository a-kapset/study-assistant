"""What model calls cost: prices read from a data file, and the count of tokens a reply reports."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

_PRICES_FILE = Path(__file__).with_name("prices.json")


@dataclass(frozen=True)
class TokenUsage:
    """The tokens of one call as the provider reports them.

    ``cached`` is the part of ``input`` served from the provider's cache, and ``reasoning`` the part of ``output``
    spent on reasoning, so neither is added on top.
    """

    input: int
    cached: int
    output: int
    reasoning: int


class Price(BaseModel):
    """US dollars per one million tokens, and the day they were compared with the provider's page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: float = Field(ge=0)
    cached_input: float = Field(ge=0)
    output: float = Field(ge=0)
    checked_on: date


def load_prices(path: Path = _PRICES_FILE) -> Mapping[str, Price]:
    """Read the price table, keyed by "provider/model"; a broken file raises, so the proxy does not start with it."""
    return TypeAdapter(dict[str, Price]).validate_json(path.read_bytes())


def call_cost(prices: Mapping[str, Price], model_ref: str, usage: TokenUsage) -> float | None:
    """Return the cost of one call in US dollars, or ``None`` when ``prices`` has no entry for ``model_ref``.

    ``None`` is not zero: a model changed in configuration without a new price line must not look free.
    """
    price = prices.get(model_ref)
    if price is None:
        return None
    uncached = usage.input - usage.cached
    return (uncached * price.input + usage.cached * price.cached_input + usage.output * price.output) / 1_000_000
