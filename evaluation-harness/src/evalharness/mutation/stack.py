"""How a mutant run starts the app again with other settings: through Docker Compose, on the stack as it runs.

The stack is started with more than one compose file (the main one and the LLM proxy's), and an app started again
with the main file only would call the model provider directly. So the files are not configured anywhere in the
harness: Docker is asked which files the running project was started with, and the same ones are used.
"""

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

# The service of the app in the stack's compose file.
_APP_SERVICE = "app"
# Longest wait until the app started again is healthy: it answers /ready once the database and its model servers do.
_WAIT_S = 180


class StackError(RuntimeError):
    """A Docker command on the stack failed; the message holds Docker's own words."""


class Stack(Protocol):
    """The running stack, as far as a mutant run changes it."""

    def recreate_app(self, settings: Mapping[str, str]) -> None:
        """Start a new container of the app with these settings on top of the stack's own, and wait until it is
        healthy; an empty mapping gives the app as the stack configures it."""
        ...


class ComposeStack:
    """The stack run by Docker Compose from the folder of its .env file."""

    def __init__(self, project: str, files: Sequence[Path], folder: Path, git_commit: str) -> None:
        """Keep the project, its compose files, the folder its .env file is in and the commit of the app's image."""
        self.project = project
        self.files = tuple(files)
        self.folder = folder
        self.git_commit = git_commit

    @classmethod
    def discover(cls, env_file: Path, git_commit: str) -> ComposeStack:
        """Find the running Compose project whose first compose file sits next to the stack's .env file.

        ``git_commit`` is the commit the app reports: the build reads it from the environment, so it is passed on
        unchanged and the image is never taken for another one.
        """
        folder = env_file.resolve().parent
        listing = _docker(["docker", "compose", "ls", "--format", "json"], folder)
        for project in json.loads(listing or "[]"):
            files = unique_files(str(project.get("ConfigFiles", "")).split(","))
            if files and os.path.normcase(files[0].parent) == os.path.normcase(folder):
                return cls(str(project["Name"]), files, folder, git_commit)
        raise StackError(f"no running Compose project was started from {folder}; start the stack first")

    def recreate_app(self, settings: Mapping[str, str]) -> None:
        """Create the app's container again with ``settings`` in the environment of Docker Compose, which wins over
        the .env file, and wait until it is healthy. Only the app changes; the other services keep running."""
        files = [arg for path in self.files for arg in ("-f", str(path))]
        command = ["docker", "compose", "-p", self.project, *files, "up", "-d", "--no-deps", "--force-recreate"]
        command += ["--wait", "--wait-timeout", str(_WAIT_S), _APP_SERVICE]
        _docker(command, self.folder, {**os.environ, "GIT_COMMIT": self.git_commit, **settings})


def unique_files(paths: Sequence[str]) -> list[Path]:
    """Return the compose files in their order, each once: Docker may list a file twice, written with another case
    of the drive letter."""
    seen: set[str] = set()
    files: list[Path] = []
    for text in paths:
        if not text.strip():
            continue
        key = os.path.normcase(os.path.abspath(text.strip()))
        if key not in seen:
            seen.add(key)
            files.append(Path(text.strip()))
    return files


def _docker(command: list[str], folder: Path, env: Mapping[str, str] | None = None) -> str:
    """Run a Docker command in ``folder`` and return what it printed; a failure raises StackError with its message."""
    result = subprocess.run(command, cwd=folder, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise StackError(f"{' '.join(command[:3])} ... failed: {(result.stderr or result.stdout).strip()[-800:]}")
    return result.stdout
