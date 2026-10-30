"""Build skin.ipynb and audio.ipynb from nb_src/*.py + common/*.py.  Usage: python3 build_notebooks.py [--check]
--check also extracts every code cell to a temp dir and runs py_compile on it (magics/shell lines are blanked)."""
import importlib.util
import json
import py_compile
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def load_cells(name):
    spec = importlib.util.spec_from_file_location(name, HERE / "nb_src" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CELLS


def to_nb(cells):
    out = []
    for kind, text in cells:
        if kind == "writefile_common":
            body = (HERE / "common" / f"{text}.py").read_text()
            kind, text = "code", f"%%writefile common/{text}.py\n{body}"
        src = text.rstrip("\n").splitlines(keepends=True)
        base = {"metadata": {}, "source": src}
        if kind == "md":
            out.append({"cell_type": "markdown", **base})
        else:
            out.append({"cell_type": "code", "execution_count": None, "outputs": [], **base})
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def check(nb, tag):
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            if src.startswith("%%writefile"):
                continue  # common/*.py are compiled separately
            lines = ["" if l.lstrip().startswith(("!", "%")) else l for l in src.splitlines()]
            f = Path(d) / f"{tag}_{i:02d}.py"
            f.write_text("\n".join(lines) + "\n")
            try:
                py_compile.compile(str(f), doraise=True)
            except py_compile.PyCompileError as e:
                bad += 1
                print(f"SYNTAX ERROR {tag} cell {i}: {e}")
        for p in sorted((HERE / "common").glob("*.py")):
            try:
                py_compile.compile(str(p), cfile=str(Path(d) / (p.name + "c")), doraise=True)
            except py_compile.PyCompileError as e:
                bad += 1
                print("SYNTAX ERROR", p, e)
    return bad


if __name__ == "__main__":
    bad = 0
    for name in ("skin", "audio"):
        nb = to_nb(load_cells(name))
        (HERE / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
        print(f"wrote {name}.ipynb ({len(nb['cells'])} cells, {n_code} code)")
        if "--check" in sys.argv:
            bad += check(nb, name)
    if "--check" in sys.argv:
        print("syntax check:", "FAILED" if bad else "OK")
    sys.exit(1 if bad else 0)
