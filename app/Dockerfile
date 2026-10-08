# Build stage: Poetry installs the app and its runtime dependencies into /app/.venv.
FROM python:3.14.8-slim-trixie AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

RUN pip install "poetry==2.5.1"

WORKDIR /app
# Dependencies first, so this layer is rebuilt only when the lock file changes, not on every code change.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root
COPY src ./src
RUN poetry install --only main

# Runtime stage: Python, the virtual environment and the code; no Poetry and no build tools.
FROM python:3.14.8-slim-trixie

ARG GIT_COMMIT=unknown
ENV APP_GIT_COMMIT=${GIT_COMMIT} \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build /app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "study_assistant.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
