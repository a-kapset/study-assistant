"""Unit tests of how a source mutant's patch is prepared: a small git repository made by each test, never the project's
own repository and never the stack."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from evalharness.mutation.source import PatchError, prepare_patch

_ORIGINAL = "def keep(score: float) -> bool:\n    return score >= 0.83\n"
_PATCH = """--- a/app/src/evalharness_code.py
+++ b/app/src/evalharness_code.py
@@ -1,2 +1,2 @@
 def keep(score: float) -> bool:
-    return score >= 0.83
+    return True
"""
_OUTSIDE = """--- a/evalharness_root.txt
+++ b/evalharness_root.txt
@@ -1 +1 @@
-root
+changed
"""


def _git(folder: Path, *args: str) -> str:
    """Run git in ``folder`` and return what it printed."""
    return subprocess.run(["git", *args], cwd=folder, capture_output=True, text=True, check=True).stdout


def _repository(tmp_path: Path) -> tuple[Path, str]:
    """Make a repository with one app code file and one file outside the app, commit them and return the folder and
    the commit."""
    repo = tmp_path / "repo"
    (repo / "app" / "src").mkdir(parents=True)
    (repo / "app" / "src" / "evalharness_code.py").write_bytes(_ORIGINAL.encode())
    (repo / "evalharness_root.txt").write_bytes(b"root\n")
    # As in the project: LF in every working tree, so a worktree holds the bytes the image was built from.
    (repo / ".gitattributes").write_bytes(b"* text=auto eol=lf\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "core.autocrlf=false", "add", ".")
    _git(repo, "-c", "user.name=evalharness", "-c", "user.email=evalharness@example.invalid", "commit", "-qm", "x")
    return repo, _git(repo, "rev-parse", "HEAD").strip()


def _patch_file(tmp_path: Path, text: str) -> Path:
    """Write a patch file next to the repository."""
    path = tmp_path / "evalharness.patch"
    path.write_bytes(text.encode())
    return path


@pytest.mark.covers("US-019", "AC-11")
def test_a_patch_is_applied_on_a_copy_of_the_commit_and_the_working_tree_is_left_as_it_was(tmp_path: Path) -> None:
    """A patch of one app code file is prepared against the commit of a small repository. The patched copy must
    hold the change, be mounted at the file's place in the image, and carry the hashes of both versions; the
    repository's own file must be unchanged, with no worktree left behind.

    A pass says that preparing a mutant never edits the code the stack is built from and that the proofs of a run
    have the right hashes to compare with. A failure points to source.py: a patch applied in the working tree would
    change the next image and every later measurement; a wrong hash would make every run invalid or, worse, valid
    without the patch.
    """
    repo, commit = _repository(tmp_path)

    prepared = prepare_patch(_patch_file(tmp_path, _PATCH), commit, repo, tmp_path / "out")

    (file,) = prepared.files
    assert file.container_path == "/app/src/evalharness_code.py", f"mounted at {file.container_path}"
    assert "return True" in file.host_file.read_text(), "the copy does not hold the patched code"
    assert file.patched_sha256 == hashlib.sha256(file.host_file.read_bytes()).hexdigest(), "wrong patched hash"
    assert file.original_sha256 == hashlib.sha256(_ORIGINAL.encode()).hexdigest(), "wrong original hash"
    assert (repo / "app" / "src" / "evalharness_code.py").read_bytes() == _ORIGINAL.encode(), "working tree changed"
    assert len(_git(repo, "worktree", "list").splitlines()) == 1, "a worktree was left behind"


@pytest.mark.covers("US-019", "AC-11")
@pytest.mark.parametrize(
    ("patch", "commit", "reason"),
    [
        pytest.param(_PATCH.replace("0.83", "0.85"), None, "apply", id="patch made for other code"),
        pytest.param(_OUTSIDE, None, "outside the app", id="patch outside the app folder"),
        pytest.param(_PATCH, "-dirty", "clean commit", id="image built from changes not committed"),
    ],
)
def test_a_patch_that_cannot_be_checked_against_the_image_is_refused_and_leaves_no_worktree(
    tmp_path: Path, patch: str, commit: str | None, reason: str
) -> None:
    """A patch is prepared that does not fit the commit, reaches outside the app's folder, or is asked for an image
    built from uncommitted changes. Each must be refused with its reason, and no worktree may be left.

    A pass says that a mutant runs only on code it was written for, inside the files the image holds. A failure
    points to the refusals in source.py: a patch that no longer fits after an app change would plant nothing, and a
    file outside the app would never reach the container, so the run would read as a defect no check sees.
    """
    repo, head = _repository(tmp_path)
    asked = head + commit if commit else head

    with pytest.raises(PatchError, match=reason):
        prepare_patch(_patch_file(tmp_path, patch), asked, repo, tmp_path / "out")

    assert len(_git(repo, "worktree", "list").splitlines()) == 1, "a worktree was left behind"
