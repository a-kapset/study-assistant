"""Source mutants: a patch of the app's code, applied on a throwaway copy of the commit the app's image was built from,
and the changed files mounted over the image when the app is started again.

The repository's own working tree is never touched: the patch is applied on a git worktree in a folder outside it,
the changed files are copied out, and the worktree is removed before any mutant runs. A patch is applied to the very
commit the running app reports, so it cannot silently change code that the image does not hold. The app's image keeps
the repository's app folder at /app, so a changed file app/src/x.py is mounted at /app/src/x.py. The hashes of each
file before and after the patch let a mutant run prove, from inside the container, which version the app runs.
"""

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# The app's folder in the repository, and where the image keeps it.
_APP_FOLDER = "app/"
_IMAGE_ROOT = "/app/"
# The service of the app in the stack's compose file.
_APP_SERVICE = "app"


class PatchError(ValueError):
    """A patch cannot be prepared; the message says which patch and why."""


@dataclass(frozen=True)
class PatchedFile:
    """One file of the app's code that a patch changes."""

    # Where the image keeps the file, e.g. /app/src/study_assistant/answer/service.py.
    container_path: str
    # The changed file, mounted over the image's own.
    host_file: Path
    patched_sha256: str
    original_sha256: str


@dataclass(frozen=True)
class PreparedPatch:
    """The files a patch changes and the compose file that mounts them over the app's image."""

    files: tuple[PatchedFile, ...]
    override: Path


def prepare_patch(patch: Path, commit: str, repo: Path, out: Path) -> PreparedPatch:
    """Apply ``patch`` on a worktree of ``commit`` in ``out``, copy the changed files to ``out`` and write the compose
    file that mounts them; the worktree is removed in every case.

    Refused with PatchError when the commit is not a clean commit of the repository (an image built from changes not
    committed, or of unknown origin, has no code a patch could be checked against), when the patch does not apply to
    that commit, or when it changes a file outside the app's folder.
    """
    if not commit or commit == "unknown" or commit.endswith("-dirty"):
        raise PatchError(f"{patch}: the app reports commit {commit!r}; a patch needs the clean commit of its image")
    worktree = out / "worktree"
    try:
        _git(repo, "worktree", "add", "--detach", str(worktree), commit)
    except subprocess.CalledProcessError as error:
        raise PatchError(f"{patch}: no worktree of commit {commit}: {error.stderr.strip()}") from error
    try:
        names = _changed_files(worktree, patch)
        outside = [name for name in names if not name.startswith(_APP_FOLDER)]
        if outside:
            raise PatchError(f"{patch}: changes files outside the app's folder: {outside}")
        originals = {name: _sha256(worktree / name) for name in names}
        _git(worktree, "apply", str(patch.resolve()))
        files = []
        for name in names:
            copy = out / "files" / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(worktree / name, copy)
            container_path = _IMAGE_ROOT + name.removeprefix(_APP_FOLDER)
            files.append(PatchedFile(container_path, copy, _sha256(copy), originals[name]))
    except subprocess.CalledProcessError as error:
        raise PatchError(f"{patch}: git {' '.join(error.cmd[1:3])} failed: {error.stderr.strip()}") from error
    finally:
        _git(repo, "worktree", "remove", "--force", str(worktree))
    override = out / "compose.mutant.json"
    override.write_text(json.dumps(_mounts(tuple(files)), indent=2), encoding="utf-8")
    return PreparedPatch(tuple(files), override)


def _changed_files(worktree: Path, patch: Path) -> list[str]:
    """Return the repository paths the patch changes, after checking that it applies to the worktree."""
    _git(worktree, "apply", "--check", str(patch.resolve()))
    listing = _git(worktree, "apply", "--numstat", str(patch.resolve()))
    names = [line.split("\t")[2] for line in listing.splitlines() if line.strip()]
    if not names:
        raise PatchError(f"{patch}: changes no file")
    return names


def _mounts(files: tuple[PatchedFile, ...]) -> dict[str, object]:
    """Return a compose file (JSON is YAML too) that mounts every changed file read-only over the image's own."""
    volumes = [
        {"type": "bind", "source": str(file.host_file.resolve()), "target": file.container_path, "read_only": True}
        for file in files
    ]
    return {"services": {_APP_SERVICE: {"volumes": volumes}}}


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(folder: Path, *args: str) -> str:
    """Run a git command in ``folder`` and return what it printed; a failure raises CalledProcessError."""
    return subprocess.run(["git", *args], cwd=folder, capture_output=True, text=True, check=True).stdout
