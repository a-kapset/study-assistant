"""API tests for the embedding space the running app computes its vectors in."""

from typing import Any, cast

import pytest

from evalharness.transport import SutTransport

# The text each model needs in front of a passage and of a query, as its model card states it. A model that is not
# listed fails the check instead of passing unchecked: look its prefixes up and add them.
# intfloat/multilingual-e5-small, model card at revision 614241f6: every input "should start with "query: " or
# "passage: "", the space included.
MODEL_PREFIXES = {
    "intfloat/multilingual-e5-small": {"passage_prefix": "passage: ", "query_prefix": "query: "},
}


@pytest.mark.covers("US-004", "AC-21")
async def test_configured_prefixes_are_the_ones_the_model_needs(sut: SutTransport) -> None:
    """The passage and query prefixes the running app works with must be exactly the ones its model needs.

    The test reads the app's effective settings from GET /v1/config and compares the two prefixes with the model
    card's, character by character. A pass tells us that the vectors in the database and the vectors of future
    queries are made from input the model was trained on. A failure names the prefix the app really got, for
    example 'passage:' without its space.

    Worth having because this mistake is silent everywhere else: an unquoted value in the .env file loses its
    trailing space on the way through Docker Compose, every vector moves a little (cosine about 0.995 to the right
    one), and ingest, the counts and readiness all stay green. The app's own reference check uses the prefixes of
    its reference file, not the ones the stack was started with.
    """
    observation = await sut.request("GET", "/v1/config")
    assert observation.status_code == 200, (
        "GET /v1/config did not answer 200, so there are no settings to compare: "
        f"{observation.transport_error or observation.body_text}"
    )
    # Only three values are read here; the shape of the whole answer is the config contract test's job.
    embedding = cast(dict[str, Any], observation.body_json)["settings"]["embedding"]
    model = embedding["model"]

    assert model in MODEL_PREFIXES, (
        f"the app runs model {model!r}, whose prefixes the harness does not know yet; add them from its model card"
    )
    configured = {"passage_prefix": embedding["passage_prefix"], "query_prefix": embedding["query_prefix"]}
    expected = MODEL_PREFIXES[model]
    assert configured == expected, f"the app runs with prefixes {configured}, but {model} needs {expected}"
