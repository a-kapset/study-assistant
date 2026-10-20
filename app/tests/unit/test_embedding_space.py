import dataclasses
import re

import pytest

from study_assistant.embed.base import EmbeddingSpace
from study_assistant.settings import EmbeddingSettings

SETTINGS = EmbeddingSettings(
    model="intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    dimension=384,
    passage_prefix="passage: ",
    query_prefix="query: ",
)
# The id of the space of SETTINGS. Vector tables are named after space ids, so a changed id
# leaves every stored vector behind in an old table; update this value only when that is intended.
GOLDEN_ID = "multilingual_e5_small_9e0ac3757316"

# A space id is a model name of at most 24 characters, "_" and 12 hex digits.
MAX_ID_LENGTH = 37


def _space() -> EmbeddingSpace:
    """The space of the test model choice."""
    return EmbeddingSpace.from_settings(SETTINGS)


@pytest.mark.covers("US-004", "AC-18")
def test_space_is_named_by_the_configured_fields() -> None:
    """Builds the space from settings and expects every field in its own place.

    Catches a mix-up such as the passage and query prefixes swapped: the id would still change with each
    setting, but passages would then be embedded with the query prefix.
    """
    assert _space() == EmbeddingSpace(
        model="intfloat/multilingual-e5-small",
        revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        passage_prefix="passage: ",
        query_prefix="query: ",
    )


@pytest.mark.covers("US-004", "AC-18")
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "intfloat/multilingual-e5-base"),
        ("revision", "0" * 40),
        ("passage_prefix", "passage:"),
        ("query_prefix", "query:"),
    ],
)
def test_any_changed_field_gives_another_space_id(field: str, value: str) -> None:
    """Changes one identity field and expects a new id.

    Catches a field left out of the id: vectors made with the old and the new value would share one table
    and be compared with each other.
    """
    changed = dataclasses.replace(_space(), **{field: value})
    assert changed.id != _space().id


def test_space_id_stays_the_same_between_runs() -> None:
    """Expects the id of a known space to be exactly the stored one.

    Catches a change of how the id is made (field order, serialization, hash length): the same configuration
    would then look like a new space, and ingest would compute every vector again into a new table.
    """
    assert _space().id == GOLDEN_ID


def test_space_id_is_a_short_plain_name() -> None:
    """Uses a long model name with capitals, dots and dashes, and expects only lower-case letters, digits and "_".

    Catches a model name copied into the id as it is: table names are built from the id, and such characters
    or a name over PostgreSQL's length limit would break the table and index statements.
    """
    space = dataclasses.replace(_space(), model="Some-Org/Multi.Lingual-Embedding-Model-Large-V2.1")
    assert re.fullmatch(r"[a-z0-9_]+", space.id)
    assert len(space.id) <= MAX_ID_LENGTH


def test_vector_table_is_named_after_the_space() -> None:
    """Expects the vector table of a known space to have exactly the stored name.

    Catches a change of how the name is made: the migration run would create a new empty table, and the vectors
    already stored would stay unused in the old one.
    """
    assert _space().vector_table == f"vectors_{GOLDEN_ID}"
