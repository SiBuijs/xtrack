#!/usr/bin/env python3
# pip install sympy
"""Symbolic commutator calculator for D (Drift), K (Kick), R (Rotate) operators.

B_x, B_y, B_z and P_z are treated as constants (no field gradients or P_z derivatives).
"""

from __future__ import annotations

from pathlib import Path

from sympy import (
    Derivative,
    Function,
    Subs,
    diff,
    expand,
    simplify,
    symbols,
    trigsimp,
)
from sympy.printing.latex import LatexPrinter

# --- Symbols -----------------------------------------------------------------

x, y, z = symbols("x y z", real=True)
Px, Py = symbols("P_x P_y", real=True)
Pz = symbols("P_z", positive=True)
q = symbols("q", real=True)

Bx, By, Bz = symbols("B_x B_y B_z", real=True)

PARTIAL_LATEX = {
    x: r"\partial_x",
    y: r"\partial_y",
    z: r"\partial_z",
    Px: r"\partial_{P_x}",
    Py: r"\partial_{P_y}",
}

class OperatorLatexPrinter(LatexPrinter):
    """Render f-derivatives as partial operators; drop explicit f."""

    def _is_f_derivative(self, expr) -> bool:
        return isinstance(expr, Derivative) and getattr(expr.expr, "func", None) and (
            expr.expr.func.__name__ == "f"
        )

    def _partial_latex(self, variables) -> str:
        from collections import Counter

        counts = Counter(variables)
        parts = []
        for var in (x, y, Px, Py):
            if var not in counts:
                continue
            name = PARTIAL_LATEX[var]
            order = counts[var]
            parts.append(name if order == 1 else f"{name}^{{{order}}}")
        return " ".join(parts) if parts else super()._print_Derivative(variables)

    def _print_Derivative(self, expr):
        if self._is_f_derivative(expr):
            return self._partial_latex(expr.variables)
        return super()._print_Derivative(expr)

    def _print_Function(self, expr, exp=None, **kwargs):
        if expr.func.__name__ == "f":
            return ""
        return super()._print_Function(expr, exp=exp)

    def _print_Subs(self, expr):
        return self._print(expr.args[0])


def latex_operators(expr) -> str:
    return OperatorLatexPrinter().doprint(expr)


OPS_NAMES = ("D", "K", "R")
FIRST_ORDER_PAIRS = (("D", "K"), ("D", "R"), ("K", "R"))


# --- Operator application ----------------------------------------------------


def apply_D(f):
    return (Px / Pz) * diff(f, x) + (Py / Pz) * diff(f, y) + diff(f, z)


def apply_K(f):
    return -q * By * diff(f, Px) + q * Bx * diff(f, Py)


def apply_R(f):
    return (q * Bz / Pz) * (Py * diff(f, Px) - Px * diff(f, Py))


OPS = {"D": apply_D, "K": apply_K, "R": apply_R}


# --- Commutator machinery ----------------------------------------------------


def _simplify_expr(expr):
    return trigsimp(simplify(expr))


def commutator(A, B, f):
    return _simplify_expr(A(B(f)) - B(A(f)))


def commutator_with_op(A, B_op, f):
    """[A, B_op] where B_op is a 1st-order commutator operator."""
    return _simplify_expr(A(B_op(f)) - B_op(A(f)))


def make_first_order_op(name_a: str, name_b: str):
    A, B = OPS[name_a], OPS[name_b]

    def op(f):
        return commutator(A, B, f)

    return op


FIRST_ORDER: dict[tuple[str, str], object] = {}


def build_first_order_cache(f):
    """Compute and cache 1st-order commutator operators."""
    FIRST_ORDER.clear()
    for name_a, name_b in FIRST_ORDER_PAIRS:
        print(f"Computing [{name_a}, {name_b}]...")
        op = make_first_order_op(name_a, name_b)
        FIRST_ORDER[(name_a, name_b)] = op
        op(f)


def _operator_view_expr(expr):
    """Normalize f-evaluation artifacts for operator-style display."""
    f_generic = Function("f")(x, y, z, Px, Py)
    expr = expr.subs(Function("f")(x, y, z, 0, 0), f_generic)

    # Remove explicit substitution wrappers on f-derivatives for display.
    for sub in list(expr.atoms(Subs)):
        base = sub.expr
        if (
            isinstance(base, Derivative)
            and getattr(base.expr, "func", None)
            and base.expr.func.__name__ == "f"
        ):
            expr = expr.subs(sub, Derivative(f_generic, *base.variables))

    return _simplify_expr(expand(expr))


# --- Naming helpers ----------------------------------------------------------


def bracket_name_1st(name_a: str, name_b: str) -> str:
    return f"[{name_a}, {name_b}]"


def bracket_name_2nd(name_a: str, name_b: str, name_c: str) -> str:
    return f"[{name_a}, [{name_b}, {name_c}]]"


# --- Computation loops -------------------------------------------------------


def compute_all_first_order(f) -> dict:
    build_first_order_cache(f)
    results = {}
    for pair in FIRST_ORDER_PAIRS:
        name = bracket_name_1st(*pair)
        expr = FIRST_ORDER[pair](f)
        results[pair] = {"expr": expr, "name": name}
    return results


def compute_all_second_order(f) -> dict:
    results = {}
    for name_a in OPS_NAMES:
        for name_b, name_c in FIRST_ORDER_PAIRS:
            name = bracket_name_2nd(name_a, name_b, name_c)
            print(f"Computing {name}...")
            inner_op = FIRST_ORDER[(name_b, name_c)]
            expr = commutator_with_op(OPS[name_a], inner_op, f)
            key = (name_a, name_b, name_c)
            results[key] = {"expr": expr, "name": name}
    return results


# --- Markdown output ---------------------------------------------------------


def to_latex_md(name: str, expr) -> str:
    rendered = latex_operators(_operator_view_expr(expr))
    return f"## {name}\n\n$$ {name} = {rendered} $$\n"


def build_markdown(parts_1st: dict, parts_2nd: dict) -> str:
    header = """# Commutators of Beam Optics Operators D, K, R

Phase space coordinates $(x, y, z, P_x, P_y)$ with $P_z$ treated as constant.
Operators (acting on a generic phase-space function):
$$D = \\frac{P_x}{P_z}\\partial_x + \\frac{P_y}{P_z}\\partial_y + \\partial_z$$
$$K = -q B_y\\partial_{P_x} + q B_x\\partial_{P_y}$$
$$R = \\frac{q B_z}{P_z}\\left(P_y\\partial_{P_x} - P_x\\partial_{P_y}\\right)$$
Field components $B_x, B_y, B_z$ are treated as constants (no $\\nabla B$ terms).
**Scope:** 1st and 2nd order commutators only (12 total).

"""
    sections = ["# First-order commutators\n"]
    for pair in FIRST_ORDER_PAIRS:
        entry = parts_1st[pair]
        sections.append(to_latex_md(entry["name"], entry["expr"]))

    sections.append("# Second-order commutators\n")
    for name_a in OPS_NAMES:
        for pair in FIRST_ORDER_PAIRS:
            key = (name_a, pair[0], pair[1])
            entry = parts_2nd[key]
            sections.append(to_latex_md(entry["name"], entry["expr"]))

    return header + "\n".join(sections)


# --- Main --------------------------------------------------------------------


def main():
    f = Function("f")(x, y, z, Px, Py)

    parts_1st = compute_all_first_order(f)
    parts_2nd = compute_all_second_order(f)

    md = build_markdown(parts_1st, parts_2nd)
    out_path = Path(__file__).parent / "commutators.md"
    out_path.write_text(md)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
