"""Structured logging: one JSON object per line in stdout.

The application factory configures it from the settings, never at import time, so the log level
always comes from the configuration the app was built with.
"""

import sys

import structlog
from structlog.tracebacks import ExceptionDictTransformer

from study_assistant.settings import LogLevel


def configure_logging(level: LogLevel) -> None:
    """Send structlog events to stdout as JSON lines and drop events below ``level``.

    structlog's configuration is process-wide: calling this again replaces it, so the most recently
    built app decides the level. Loggers are not cached, so a new configuration applies at once.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            # The traceback as data, without the values of local variables: those hold the learner's question,
            # the course passages and the model's reply, which the logs must not keep.
            structlog.processors.ExceptionRenderer(ExceptionDictTransformer(show_locals=False)),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )
