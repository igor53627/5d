"""The line and function length limits of docs/style-profile.md, enforced as tests."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE_COLUMNS_MAX = 100
FUNCTION_LINES_MAX = 70
SKIPPED_DIRECTORY_NAMES = frozenset({"__pycache__", "build", "dist", "env", "venv"})
LEGACY_LIST_HINT = "the legacy list is `extend-exclude` under [tool.ruff] in pyproject.toml"


def legacy_files() -> list:
    """Read the legacy list, one quoted path on each line, from pyproject.toml.

    The list is the automated mirror of exceptions E1-E5 in docs/style-profile.md and
    has one copy, which ruff reads too. A regular expression reads it because `tomllib`
    does not exist before Python 3.11.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^extend-exclude = \[\n(.*?)^\]", text, re.DOTALL | re.MULTILINE)
    assert block is not None, LEGACY_LIST_HINT
    entries = [line.strip() for line in block.group(1).splitlines()]
    entries = [line for line in entries if line and not line.startswith("#")]
    names = [re.fullmatch(r'"([^"]+)",\s*(?:#.*)?', line) for line in entries]
    unreadable = [line for line, name in zip(entries, names) if name is None]
    assert not unreadable, f"cannot read legacy list entries: {unreadable}"
    return [name.group(1) for name in names]


def python_files() -> list:
    """Return the repository path of every Python file, tracked or new, in sorted order.

    Hidden directories, virtual environments and build output are not part of the code.
    """
    names = []
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT)
        directories = relative.parts[:-1]
        hidden = any(part.startswith(".") for part in directories)
        skipped = any(
            part in SKIPPED_DIRECTORY_NAMES or part.endswith(".egg-info") for part in directories
        )
        if not hidden and not skipped:
            names.append(relative.as_posix())
    return names


def violations(text: str, name: str) -> list:
    """Return one message for each line and each function of the text that is too long."""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if len(line) > LINE_COLUMNS_MAX:
            found.append(f"{name}:{number}: line has {len(line)} columns")
    for node in ast.walk(ast.parse(text, filename=name)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            line_count = node.end_lineno - node.lineno + 1
            if line_count > FUNCTION_LINES_MAX:
                found.append(f"{name}:{node.lineno}: function {node.name} has {line_count} lines")
    return found


def violations_of_file(name: str) -> list:
    """Return the violations of one file, given by its repository path."""
    return violations((ROOT / name).read_text(encoding="utf-8"), name)


def function_source(body_line_count: int) -> str:
    """Return the source of a function that spans one `def` line and the given body."""
    return "def example():\n" + "    pass\n" * body_line_count


def test_checker_accepts_the_limit_and_rejects_one_more() -> None:
    """Goal: the checker itself works. Method: synthetic text at each limit and one past it."""
    assert violations("#" * LINE_COLUMNS_MAX + "\n", "at_limit.py") == []
    assert violations("#" * (LINE_COLUMNS_MAX + 1) + "\n", "over.py") == [
        "over.py:1: line has 101 columns"
    ]
    assert violations(function_source(FUNCTION_LINES_MAX - 1), "at_limit.py") == []
    assert violations(function_source(FUNCTION_LINES_MAX), "over.py") == [
        "over.py:1: function example has 71 lines"
    ]


def test_legacy_list_names_plain_files_once() -> None:
    """Goal: each suppression has the scope of one file. Method: reject patterns and repeats."""
    legacy = legacy_files()
    assert legacy, LEGACY_LIST_HINT
    assert len(legacy) == len(set(legacy))
    for name in legacy:
        assert name.endswith(".py"), name
        assert not set(name) & set("*?[]{}!"), name


def test_files_outside_the_legacy_list_meet_the_limits() -> None:
    """Goal: new code obeys 100 columns and 70 lines. Method: check every other file."""
    legacy = set(legacy_files())
    checked = [name for name in python_files() if name not in legacy]
    assert "src/helpers.py" in checked
    found = [message for name in checked for message in violations_of_file(name)]
    limits = f"limits: {LINE_COLUMNS_MAX} columns, {FUNCTION_LINES_MAX} lines per function"
    assert not found, "\n".join([limits, *found])


def test_legacy_list_only_shrinks() -> None:
    """Goal: no stale exception. Method: each legacy file must exist and exceed a limit.

    When this fails, remove the named file from the legacy list and correct the count
    in docs/style-profile.md. Never add a file to the list.
    """
    stale = []
    for name in legacy_files():
        if not (ROOT / name).is_file():
            stale.append(f"{name}: no longer exists")
        elif not violations_of_file(name):
            stale.append(f"{name}: meets the limits now")
    assert not stale, "\n".join([LEGACY_LIST_HINT, *stale])
