"""Normalize Python comments in memory while preserving program structure."""
import ast
from io import StringIO
import tokenize


def same_ast(left, right):
    """Compare syntax fields iteratively, ignoring source locations."""
    pending = [(left, right)]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if isinstance(left, ast.AST):
            pending.extend((getattr(left, name, None), getattr(right, name, None)) for name in left._fields)
        elif isinstance(left, list):
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right))
        elif left != right:
            return False
    return True


def strip_comments(program, *, reject_docstrings=False):
    """Normalize comments without changing the AST, optionally screening docstrings."""
    before = ast.parse(program.encode("utf-8"))
    if reject_docstrings and any(
        ast.get_docstring(node, clean=False) is not None for node in ast.walk(before)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        raise ValueError("Program contains a docstring")
    lines = StringIO(program).readlines()
    protected, comments = set(), {}
    for token in tokenize.generate_tokens(StringIO(program).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0] - 1] = token.start[1]
        if token.type in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)):
            protected.update(range(token.start[0] - 1, token.end[0] - 1))
    result = []
    for index, line in enumerate(lines):
        ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        body = line[:-len(ending)] if ending else line
        if index in comments:
            body = body[:comments[index]]
            if not body.strip():
                continue
        result.append((body if index in protected else body.rstrip(" \t\f")) + ending)
    normalized = "".join(result)
    if not same_ast(before, ast.parse(normalized.encode("utf-8"))):
        raise ValueError("Comment removal changed the program AST")
    return normalized
