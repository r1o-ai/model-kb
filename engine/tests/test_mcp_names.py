"""Every name the model-kb MCP server uses must be defined or imported.

Regression for 2026-10-31: a refactor dropped `recipe_to_serve_request` from the
import line while model_get and _weights_present still called it. Python only
raises NameError when the call runs, so model_get failed on every call and
family-name model_serve crashed, with the server otherwise healthy.

Static (ast) so it needs no fastmcp, no records and no live data.

    python3 -I -m unittest discover -s engine/tests -v
"""
from __future__ import annotations

import ast
import builtins
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
SERVERS = sorted(ENGINE.glob("*-mcp.py"))


def _bound_in(node: ast.AST) -> set[str]:
    """Names a scope binds: params, assignments, imports, defs, loop/with/except/comprehension targets."""
    names: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            names.add(arg.arg)
    body = node.body if isinstance(node.body, list) else [node.body]
    stack = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
            stack.extend(n.decorator_list)
            continue  # its body is its own scope
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                names.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
        elif isinstance(n, ast.MatchAs) and n.name:
            names.add(n.name)
        stack.extend(ast.iter_child_nodes(n))
    return names


def undefined_names(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    module_names = _bound_in(tree) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    missing: list[tuple[int, str]] = []

    def visit(scope: ast.AST, visible: set[str]) -> None:
        here = visible | (_bound_in(scope) if scope is not tree else set())
        for child in _child_scopes(scope):
            visit(child, here)  # nested scopes see the enclosing scope's names (closures)
        for n in _own_nodes(scope):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in here:
                missing.append((n.lineno, n.id))

    visit(tree, module_names)
    return sorted(set(missing))


def _child_scopes(scope: ast.AST):
    """Function/lambda scopes directly nested in this scope (not deeper ones)."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield n
            continue
        stack.extend(ast.iter_child_nodes(n))


def _own_nodes(scope: ast.AST):
    """Nodes in this scope, not inside nested function/lambda scopes."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(getattr(n, "decorator_list", []))
            stack.extend(n.args.defaults + [d for d in n.args.kw_defaults if d])
            continue
        yield n
        stack.extend(ast.iter_child_nodes(n))


class McpServerNames(unittest.TestCase):
    def test_servers_found(self):
        self.assertTrue(SERVERS, f"no *-mcp.py under {ENGINE}")

    def test_no_undefined_names(self):
        for path in SERVERS:
            with self.subTest(server=path.name):
                missing = undefined_names(path.read_text())
                self.assertEqual(missing, [], f"{path.name}: names used but never defined/imported: {missing}")

    def test_detector_catches_the_2026_10_31_regression(self):
        src = (
            "from ingest_serve_configs import asmi_body_of\n"
            "def model_get(rec):\n"
            "    return recipe_to_serve_request(rec)\n"
        )
        self.assertEqual(undefined_names(src), [(3, "recipe_to_serve_request")])

    def test_detector_accepts_locals_closures_and_comprehensions(self):
        src = (
            "import json\n"
            "def f(a, *rest, k=1, **kw):\n"
            "    x = [y for y in rest if y]\n"
            "    try:\n"
            "        pass\n"
            "    except ValueError as e:\n"
            "        x.append(e)\n"
            "    def g():\n"
            "        return a + k + len(x)\n"
            "    return json.dumps({kk: vv for kk, vv in kw.items()}), g\n"
        )
        self.assertEqual(undefined_names(src), [])


if __name__ == "__main__":
    unittest.main()
