"""Request traces with OpenTelemetry: where finished spans go, and one span per stage of our own code.

The exporter is configured by the standard ``OTEL_*`` environment variables (for example
``OTEL_EXPORTER_OTLP_ENDPOINT``, or ``OTEL_SDK_DISABLED=true`` to switch tracing off), so a different trace
backend needs no code change. Nothing is built at import time: the server's entry point builds the provider.
"""

from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, Tracer

# The name the app's own spans are filed under in a trace, next to the names of the instrumentation libraries.
TRACER_NAME = "study_assistant"


def build_tracer_provider(service_name: str, service_version: str) -> TracerProvider:
    """Return a provider that sends finished spans in batches over OTLP to the address the environment names.

    Spans are sent from a background thread, so a trace backend that is down slows no request; the batch still
    waiting when the server stops is sent by ``shutdown``.
    """
    resource = Resource.create({SERVICE_NAME: service_name, SERVICE_VERSION: service_version})
    # The app shuts the provider down when the server stops; a second shutdown at interpreter exit would only warn.
    provider = TracerProvider(resource=resource, shutdown_on_exit=False)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return provider


@contextmanager
def stage_span(tracer: Tracer, name: str) -> Generator[Span]:
    """Wrap one stage of a request in a span whose status is ERROR when the stage raises, then re-raise.

    Only the exception's class is written to the span, as the ``error.type`` attribute and the status text:
    an exception message or its chained causes can hold the learner's question or the model's reply, and a
    trace is kept and shown outside the request.
    """
    with tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
        try:
            yield span
        except Exception as error:
            error_type = type(error).__name__
            span.set_attribute("error.type", error_type)
            span.set_status(Status(StatusCode.ERROR, error_type))
            raise
