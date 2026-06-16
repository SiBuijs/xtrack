#!/usr/bin/env python3
# pip install sympy
"""Symbolic commutator calculator for D (Drift), K (Kick), R (Rotate) operators."""

from __future__ import annotations

from pathlib import Path

from sympy import (
    Add,
    Derivative,
    Function,
    S,
    Subs,
    diff,
    expand,
    powsimp,
    simplify,
    sqrt,
    symbols,
    trigsimp,
)
from sympy.printing.latex import LatexPrinter

# --- Symbols -----------------------------------------------------------------

x, y, z = symbols("x y z", real=True)
Px, Py = symbols("P_x P_y", real=True)
P = symbols("P", positive=True)
q = symbols("q", real=True)

Pz = sqrt(P**2 - Px**2 - Py**2)
PZ2 = P**2 - Px**2 - Py**2

Bx = Function("B_x")(x, y)
By = Function("B_y")(x, y)
Bz = Function("B_z")(x, y)

PARTIAL_LATEX = {
    x: r"\partial_x",
    y: r"\partial_y",
    z: r"\partial_z",
    Px: r"\partial_{P_x}",
    Py: r"\partial_{P_y}",
}

B_FIELD_NAMES = {"B_x", "B_y", "B_z"}


class OperatorLatexPrinter(LatexPrinter):
    """Render f-derivatives as partial operators; drop explicit f and B(x,y)."""

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
        if getattr(expr.expr, "func", None) and expr.expr.func.__name__ in B_FIELD_NAMES:
            bname = f"B_{{{expr.expr.func.__name__.split('_')[1]}}}"
            return f"{self._partial_latex(expr.variables)} {bname}"
        return super()._print_Derivative(expr)

    def _print_Function(self, expr, exp=None, **kwargs):
        name = expr.func.__name__
        if name in B_FIELD_NAMES:
            tex = f"B_{{{name.split('_')[1]}}}"
            if exp is not None:
                return f"{tex}^{{{exp}}}"
            return tex
        if name == "f":
            return ""
        return super()._print_Function(expr, exp=exp)

    def _print_Subs(self, expr):
        return self._print(expr.args[0])

    def _is_pz_power(self, expr) -> bool:
        return expr.is_Pow and expand(expr.base) == expand(PZ2)

    def _print_Pow(self, expr):
        if self._is_pz_power(expr):
            exp = expr.exp
            if exp == S.Half:
                return "P_z"
            if exp == -S.Half:
                return "P_z^{-1}"
            if exp == S(3) / 2:
                return "P_z^{3}"
            if exp == -S(3) / 2:
                return "P_z^{-3}"
            if exp == 2:
                return "P_z^{2}"
            if exp == -1:
                return "P_z^{-1}"
            if exp == -2:
                return "P_z^{-2}"
            return f"P_z^{{{self._print(exp)}}}"
        return super()._print_Pow(expr)


def latex_operators(expr) -> str:
    expr = expr.subs(sqrt(PZ2), Pz)
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
    expr = simplify(expr)
    if expr.has(sqrt):
        expr = powsimp(expr, force=True)
    return trigsimp(expr)


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


# --- Term splitting ----------------------------------------------------------


def _has_field_gradient(term) -> bool:
    for d in term.atoms(Derivative):
        if getattr(d.expr, "func", None) and d.expr.func.__name__ in (
            "B_x",
            "B_y",
            "B_z",
        ):
            return True
    return False


def _recombine(terms):
    if not terms:
        return S.Zero
    return _simplify_expr(Add(*terms))


def _tidy_paraxial(expr):
    """Extra cleanup for _1 parts to expose algebraic cancellations."""
    return _simplify_expr(expand(expr))


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


def _restore_pz(expr):
    """After Px=Py=0, SymPy writes P_z as P; put P_z back in denominators."""
    return _simplify_expr(expr.subs(P, Pz))


def _expand_subs(expr):
    for sub in expr.atoms(Subs):
        expr = expr.subs(sub, sub.doit())
    return _simplify_expr(expr)


def split_into_parts(expr):
    """Split expr into _0, _1, gradB buckets.

    Field-gradient terms go to gradB. The remainder is split into leading order
    (_0, evaluated at Px=Py=0 but keeping P_z) and paraxial correction (_1).
    """
    expr = expand(_simplify_expr(expr))
    if expr == 0:
        return {"0": S.Zero, "1": S.Zero, "gradB": S.Zero}

    terms_gradB, terms_rest = [], []
    for term in Add.make_args(expr):
        if _has_field_gradient(term):
            terms_gradB.append(term)
        else:
            terms_rest.append(term)

    rest = _recombine(terms_rest)
    gradB = _recombine(terms_gradB)
    leading = _expand_subs(_restore_pz(rest.subs({Px: 0, Py: 0})))
    paraxial = _simplify_expr(rest - leading)

    return {"0": leading, "1": paraxial, "gradB": gradB}


# --- Verification ------------------------------------------------------------


def expected_0_exprs(f):
    return {
        ("D", "K"): (q * By / Pz) * diff(f, x) - (q * Bx / Pz) * diff(f, y),
        ("D", "R"): -(q * Bz / Pz) * (Py / Pz) * diff(f, x)
        + (q * Bz / Pz) * (Px / Pz) * diff(f, y),
        ("K", "R"): (q**2 * Bz / Pz) * (Bx * diff(f, Px) + By * diff(f, Py)),
    }


def _normalize_test_function(expr, f):
    """Leading-order parts may evaluate f at Px=Py=0; compare in generic form."""
    f_at_origin = Function("f")(x, y, z, 0, 0)
    return _expand_subs(expr.subs(f_at_origin, f))


def verify_first_order(parts_1st: dict, f, warnings: list[str]) -> None:
    """Verify [D,K]_0 against the analytic leading-order result."""
    pair = ("D", "K")
    expected = _simplify_expr(expected_0_exprs(f)[pair])
    computed = _normalize_test_function(parts_1st[pair]["0"], f)
    diff_expr = _simplify_expr(computed - expected)
    if not diff_expr.equals(0):
        msg = (
            f"WARNING: [{pair[0]}, {pair[1]}]_0 mismatch!\n"
            f"  Expected: {latex_operators(expected)}\n"
            f"  Actual:   {latex_operators(computed)}\n"
            f"  Diff:     {latex_operators(diff_expr)}"
        )
        print(msg)
        warnings.append(msg)


# --- Naming helpers ----------------------------------------------------------


def bracket_name_1st(name_a: str, name_b: str) -> str:
    return f"[{name_a}, {name_b}]"


def bracket_name_2nd(name_a: str, name_b: str, name_c: str) -> str:
    return f"[{name_a}, [{name_b}, {name_c}]]"


def parts_suffix(name: str) -> str:
    """LaTeX-safe subscript label for [D,K] -> D,K."""
    return name.replace("[", "").replace("]", "").replace(", ", ",").replace(" ", "")


# --- Computation loops -------------------------------------------------------


def compute_all_first_order(f) -> dict:
    build_first_order_cache(f)
    results = {}
    for pair in FIRST_ORDER_PAIRS:
        name = bracket_name_1st(*pair)
        expr = FIRST_ORDER[pair](f)
        parts = split_into_parts(expr)
        results[pair] = {"full": expr, "parts": parts, "name": name}
    return results


def compute_all_second_order(f) -> dict:
    results = {}
    for name_a in OPS_NAMES:
        for name_b, name_c in FIRST_ORDER_PAIRS:
            name = bracket_name_2nd(name_a, name_b, name_c)
            print(f"Computing {name}...")
            inner_op = FIRST_ORDER[(name_b, name_c)]
            expr = commutator_with_op(OPS[name_a], inner_op, f)
            parts = split_into_parts(expr)
            key = (name_a, name_b, name_c)
            results[key] = {"full": expr, "parts": parts, "name": name}
    return results


# --- Markdown output ---------------------------------------------------------


def to_latex_md(name: str, parts: dict, warning: str | None = None) -> str:
    suffix = parts_suffix(name)
    # Use an extra simplification pass for the paraxial piece to expose
    # cancellations like those in [D,K]_1.
    leading = _operator_view_expr(parts["0"])
    paraxial = _tidy_paraxial(_operator_view_expr(parts["1"]))
    gradb = _operator_view_expr(parts["gradB"])
    lines = [f"## {name}", ""]
    if warning:
        lines.append(f"**Warning:** {warning}")
        lines.append("")
    lines.extend(
        [
            f"$$ {name} = [{suffix}]_0 + [{suffix}]_1 + [{suffix}]_{{\\nabla B}} $$",
            "",
            f"**Leading order** ($[{suffix}]_0$, always nonzero):",
            "",
            f"$$ [{suffix}]_0 = {latex_operators(leading)} $$",
            "",
            f"**Paraxial correction** ($[{suffix}]_1$, vanishes for $P_\\perp \\ll P_z$):",
            "",
            f"$$ [{suffix}]_1 = {latex_operators(paraxial)} $$",
            "",
            f"**Field gradient term** ($[{suffix}]_{{\\nabla B}}$, vanishes for uniform field):",
            "",
            f"$$ [{suffix}]_{{\\nabla B}} = {latex_operators(gradb)} $$",
            "",
        ]
    )
    return "\n".join(lines)


def build_markdown(
    parts_1st: dict,
    parts_2nd: dict,
    warnings: list[str],
) -> str:
    header = """# Commutators of Beam Optics Operators D, K, R

Phase space coordinates $(x, y, z, P_x, P_y)$ with dependent momentum
$P_z = \\sqrt{P^2 - P_x^2 - P_y^2}$ ($P$ fixed).

Operators (acting on a generic phase-space function):

$$D = \\frac{P_x}{P_z}\\partial_x + \\frac{P_y}{P_z}\\partial_y + \\partial_z$$

$$K = -q B_y\\partial_{P_x} + q B_x\\partial_{P_y}$$

$$R = \\frac{q B_z}{P_z}\\left(P_y\\partial_{P_x} - P_x\\partial_{P_y}\\right)$$

Field components $B_x, B_y, B_z$ depend on $(x, y)$ only.

**Scope:** 1st and 2nd order commutators only (12 total).

"""
    if warnings:
        header += "**Verification warnings:**\n\n"
        for w in warnings:
            header += f"- {w}\n"
        header += "\n"

    sections = ["# First-order commutators\n"]
    for pair in FIRST_ORDER_PAIRS:
        entry = parts_1st[pair]
        warn = next((w for w in warnings if bracket_name_1st(*pair) in w), None)
        sections.append(to_latex_md(entry["name"], entry["parts"], warn))

    sections.append("# Second-order commutators\n")
    for name_a in OPS_NAMES:
        for pair in FIRST_ORDER_PAIRS:
            key = (name_a, pair[0], pair[1])
            entry = parts_2nd[key]
            sections.append(to_latex_md(entry["name"], entry["parts"]))

    return header + "\n".join(sections)


# --- Main --------------------------------------------------------------------


def main():
    f = Function("f")(x, y, z, Px, Py)
    warnings: list[str] = []

    parts_1st_raw = compute_all_first_order(f)
    parts_1st = {pair: entry["parts"] for pair, entry in parts_1st_raw.items()}
    verify_first_order(parts_1st, f, warnings)

    for pair, entry in parts_1st_raw.items():
        total = _simplify_expr(
            entry["parts"]["0"] + entry["parts"]["1"] + entry["parts"]["gradB"]
        )
        if _simplify_expr(total - entry["full"]) != 0:
            print(f"WARNING: split mismatch for [{pair[0]}, {pair[1]}]")

    parts_2nd = compute_all_second_order(f)

    for entry in parts_2nd.values():
        total = _simplify_expr(
            entry["parts"]["0"] + entry["parts"]["1"] + entry["parts"]["gradB"]
        )
        if _simplify_expr(total - entry["full"]) != 0:
            print(f"WARNING: split mismatch for {entry['name']}")

    md = build_markdown(parts_1st_raw, parts_2nd, warnings)
    out_path = Path(__file__).parent / "commutators.md"
    out_path.write_text(md)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
