# Build stage: Poetry installs the harness and its runtime dependencies into /proxy/.venv.
FROM python:3.14.8-slim-trixie AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

RUN pip install "poetry==2.5.1"

WORKDIR /proxy
# Dependencies first, so this layer is rebuilt only when the lock file changes, not on every code change.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root
COPY src ./src
RUN poetry install --only main

# Runtime stage: Python, the virtual environment and the code; no Poetry and no build tools.
FROM python:3.14.8-slim-trixie

ENV PATH="/proxy/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --system --uid 10001 --no-create-home harness
WORKDIR /proxy
COPY --from=build /proxy /proxy
USER harness

EXPOSE 8100
CMD ["python", "-m", "evalharness.proxy"]
