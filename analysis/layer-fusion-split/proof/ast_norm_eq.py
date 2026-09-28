"""AST evidence for a formatting-only commit: compare each path's blob at OLD with its blob at NEW.

Usage: python3 ast_norm_eq.py <old-rev> <new-rev> <path>...

"exact" compares the two ASTs as they are. "normalized" first parses string annotations and compares the
top-level imports as a sorted multiset, every other statement in order. Both sides are read with `git show`,
so the result does not depend on what is checked out.
"""

import ast
import subprocess
import sys


class Unquote(ast.NodeTransformer):
    def _fix(self, node):
        return (
            ast.parse(node.value, mode="eval").body
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            else node
        )

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.returns = node.returns and self._fix(node.returns)
        return node

    def visit_arg(self, node):
        node.annotation = node.annotation and self._fix(node.annotation)
        return node


def norm(src):
    tree = Unquote().visit(ast.parse(src))
    imports = sorted(
        ast.dump(s) for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))
    )
    rest = [
        ast.dump(s)
        for s in tree.body
        if not isinstance(s, (ast.Import, ast.ImportFrom))
    ]
    return imports, rest


def blob(rev, path):
    return subprocess.run(
        ["git", "show", f"{rev}:{path}"], capture_output=True, text=True, check=True
    ).stdout


old_rev, new_rev = sys.argv[1], sys.argv[2]
for p in sys.argv[3:]:
    old, new = blob(old_rev, p), blob(new_rev, p)
    exact = ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))
    print(
        p,
        "exact:",
        "identical" if exact else "DIFFERENT",
        "| normalized:",
        "identical" if norm(old) == norm(new) else "DIFFERENT",
    )
