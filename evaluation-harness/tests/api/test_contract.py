"""Contract tests of the running app, generated from the API description the app itself publishes.

Schemathesis reads ``/openapi.json`` and makes one test per operation. Each test sends many generated requests,
valid ones and ones that break the declared rules on purpose, and checks every answer against the description:
a status the operation declares, a body of the declared shape, never a server error.

Three operations are left out. Asking a question calls the answer model, ingesting a course rewrites its stored
corpus, and registering a learner would leave a new learner behind with every request.
"""

from collections.abc import Iterator
from typing import Any

import pytest
import schemathesis
from schemathesis.specs.openapi.schemas import OpenApiSchema

from evalharness.exam_client import ADMIN_KEY_HEADER, LEARNERS_PATH, ExamClient
from evalharness.settings import HarnessSettings, StackSettings

# A generated request to one operation; the library's own type arguments are of no interest here.
type ApiCase = schemathesis.Case[schemathesis.APIOperation[Any, Any, Any, Any]]

# Never sent by this test: asking calls the answer model, ingesting rewrites a stored corpus, and registering
# leaves a learner behind with every request.
_ASK = "/v1/ask"
_INGEST = r"^/v1/admin/courses/[^/]+/ingest$"
_LEFT_OUT = {("POST", _ASK), ("POST", "/v1/admin/courses/{course_id}/ingest"), ("POST", LEARNERS_PATH)}


@pytest.fixture
def api_schema(settings: HarnessSettings) -> OpenApiSchema:
    """Load the API description from the running app; a request for it that fails stops the test with the
    reason."""
    return schemathesis.openapi.from_url(f"{str(settings.sut_base_url).rstrip('/')}/openapi.json")


@pytest.fixture
def caller_headers(settings: HarnessSettings, stack: StackSettings) -> Iterator[dict[str, str]]:
    """The headers of a caller that may use every endpoint: the admin key and the key of a learner registered
    for this test and removed after it."""
    admin_key = stack.admin_api_key.get_secret_value()
    with ExamClient(settings, admin_key) as client:
        learner = client.register_learner()
        try:
            yield {ADMIN_KEY_HEADER: admin_key, **learner.headers}
        finally:
            client.remove_learner(learner)


# The operations are filtered here, on the schema the tests are made from: a filter set inside the fixture is
# not applied, and every operation would be called.
schema = (
    schemathesis.pytest.from_fixture("api_schema")
    .exclude(path=_ASK)
    .exclude(path_regex=_INGEST)
    .exclude(path=LEARNERS_PATH, method="POST")
)


@pytest.mark.covers("US-020", "AC-13")
@schema.parametrize()
def test_every_answer_is_what_the_api_description_declares(case: ApiCase, caller_headers: dict[str, str]) -> None:
    """Sends generated requests to one operation of the running app, as a caller with both keys, and checks every
    answer against the app's own API description.

    What it tells: a pass says a client written from the published description meets no status, body shape or
    server error the description does not announce, for valid and for deliberately invalid input. A failure names
    the operation, shows the request as a curl command and says which declared rule the answer broke.

    Why it is worth having: the app's own tests compare answers with the models in its code, so a status that the
    code returns but the description does not list, or generated input that reaches the database and ends in a
    500, passes there unseen. Only the description and the running app together show it.
    """
    operation = (case.method.upper(), case.path)
    assert operation not in _LEFT_OUT, f"{operation} must never be called by this test; its filter does not work"
    case.call_and_validate(headers=caller_headers)
