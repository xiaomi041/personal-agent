# -*- coding: utf-8 -*-
"""静态检查：找出模块里引用了但没定义的全局名（按字节码 LOAD_GLOBAL，含嵌套函数）。一次性工具。"""
import builtins
import dis
import importlib
import sys
import types


def walk_code(code):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from walk_code(const)


def globals_used(code):
    names = set()
    for ins in dis.get_instructions(code):
        if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME", "STORE_GLOBAL", "DELETE_GLOBAL"):
            arg = ins.argval
            if isinstance(arg, str):
                names.add(arg)
    return names


def check(modname):
    mod = importlib.import_module(modname)
    g = mod.__dict__
    missing = {}
    for obj in list(g.values()):
        if not isinstance(obj, types.FunctionType):
            continue
        if getattr(obj, "__module__", None) != modname:
            continue
        for code in walk_code(obj.__code__):
            for name in globals_used(code):
                if name in g or hasattr(builtins, name):
                    continue
                missing.setdefault(name, set()).add(code.co_name)
    return missing


if __name__ == "__main__":
    bad = 0
    for m in sys.argv[1:]:
        miss = check(m)
        if miss:
            bad += 1
            print(f"### {m}")
            for name, fns in sorted(miss.items()):
                print(f"  {name}: 用于 {', '.join(sorted(fns))}")
    print("OK" if not bad else f"{bad} 个模块有问题")
