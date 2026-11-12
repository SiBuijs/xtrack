"""Note 16, step 6: per-side integrals int_0^inf of B_S and A_1..A_4
(b^{(k)}(0) = b0_k, Phi_p = int_0^inf b). Run: python 16b_half_line_integrals.py
"""
import sympy as sp
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_s = import_module('16a_tilted_solenoid_series')
A, d, S, th, N = _s.A, _s.d, _s.S, _s.th, _s.N
P = sp.Symbol('Phi_p')               # int_0^inf b
b0 = sp.symbols('b0_0:12')           # b^{(k)}(0)
def half_int(e):
    # int_0^inf of polynomial in S times d_k(S); d_k = b^{(k-1)}
    e = sp.expand(e); out = 0
    for term in e.as_ordered_terms():
        n = sp.degree(term, S)
        ks = [k for k in range(1, N) if term.has(d[k])]
        assert len(ks) == 1, term
        k = ks[0]; coef = term/(S**n*d[k])
        bo = k-1   # b^{(bo)}
        if bo == n: val = (-1)**n*sp.factorial(n)*P
        elif bo > n: val = (-1)**n*sp.factorial(n)*(-b0[bo-n-1])
        else: raise ValueError(term)
        out += coef*val
    return sp.expand(out)
def dS(e):
    return sp.expand(sp.diff(e, S) + sum(sp.diff(e, d[n])*d[n+1] for n in range(N-1)))
print("int_0^inf B_S:", half_int(dS(A(0))) )
for m in range(1, 5):
    print(f"int_0^inf A_{m}:", sp.collect(half_int(A(m)), th))
