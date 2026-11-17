"""Resolve package sources, subprocess commands and fresh-worker import paths."""

import ast
import os
import sys
from pathlib import Path

LLM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = LLM_ROOT.parent


def source_path(name):
    """Find a uniquely named implementation without importing its dependencies."""
    filename = Path(name).name
    if not filename.endswith(".py"):
        filename += ".py"
    matches = sorted(LLM_ROOT.glob(f"*/{filename}"))
    if len(matches) != 1:
        raise ValueError(f"Expected one LLM source for {name!r}, found {len(matches)}")
    return matches[0]


def module_command(name, *arguments, python=None):
    """Build a package command; the child environment needs project_pythonpath()."""
    path = source_path(name).relative_to(REPOSITORY_ROOT).with_suffix("")
    module = ".".join(path.parts)
    return [python or sys.executable, "-B", "-u", "-m", module, *map(str, arguments)]


def project_pythonpath(existing=""):
    """Make the package available to fresh workers, including after a cwd change."""
    paths = [str(REPOSITORY_ROOT)] + existing.split(os.pathsep)
    return os.pathsep.join(dict.fromkeys(path for path in paths if path))


def code_dependencies(paths):
    """Freeze package imports and the CLI without loading TVM or model libraries.

    Manifests keep full paths. Frozen copies use basenames, so implementation
    filenames in the dependency closure must remain unique.
    """
    pending = [Path(path).resolve() for path in paths] + [LLM_ROOT / "__main__.py"]
    found = {}
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found[path] = None
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("LLM."):
                names = [node.module] + [node.module + "." + item.name for item in node.names]
            elif isinstance(node, ast.Import):
                names = [item.name for item in node.names if item.name.startswith("LLM.")]
            for name in names:
                dependency = REPOSITORY_ROOT.joinpath(*name.split(".")).with_suffix(".py")
                if dependency.is_file() and dependency not in found:
                    pending.append(dependency)
    return sorted(found)
