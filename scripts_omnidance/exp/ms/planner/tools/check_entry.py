#!/usr/bin/env python3
"""AST check that a Nebula entry keeps the helper return contracts.

Usage: check_entry.py entry.py [entry.py ...]
"""
import ast
import sys
from pathlib import Path


def is_dict_call(node):
    return isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'dict'


def check(path):
    tree = ast.parse(Path(path).read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    problems = []
    if 'train_command' in funcs:
        for ret in [n for n in ast.walk(funcs['train_command']) if isinstance(n, ast.Return) and n.value]:
            if not (isinstance(ret.value, ast.Tuple) and len(ret.value.elts) == 2):
                problems.append('train_command returns non 2-tuple at line %d' % ret.lineno)
    # mode whitelist must cover the mode declared in mode.txt next to the entry
    mode_file = Path(path).parent / 'mode.txt'
    if mode_file.is_file():
        declared = mode_file.read_text().strip()
        literals = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Str):            # py<3.8
                literals.add(node.s)
            elif node.__class__.__name__ == 'Constant' and isinstance(getattr(node, 'value', None), str):
                literals.add(node.value)
        if declared and declared not in literals:
            problems.append('mode.txt %r missing from entry whitelist' % declared)
    if 'data_config' in funcs:
        for ret in [n for n in ast.walk(funcs['data_config']) if isinstance(n, ast.Return) and n.value]:
            if not is_dict_call(ret.value):
                problems.append('data_config returns non-dict at line %d' % ret.lineno)
    return problems


def main():
    failed = False
    for arg in sys.argv[1:]:
        problems = check(arg)
        print('%-48s %s' % (arg, 'OK' if not problems else problems))
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
