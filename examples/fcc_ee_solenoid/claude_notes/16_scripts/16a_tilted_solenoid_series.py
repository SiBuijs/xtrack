"""Note 16, step 4: theta expansion of the tilted-solenoid beam-frame
coefficients A_0..A_4 and of the true coefficients T_1..T_4.

d_n stands for a^{(n)}(S), with a' = b (on-axis field). Run: python 16a_tilted_solenoid_series.py
"""
import sympy as sp
S, th = sp.symbols('S theta', real=True)
N = 16
d = sp.symbols('d0:%d' % N)  # d_n = a^{(n)}(S); d1=b, d2=b', ...
c, s = sp.cos(th), sp.sin(th)
ORDER = 6
def a_der_at_Sc(n):
    # a^{(n)}(S cos th) Taylor-expanded about S
    h = S*(c-1)
    return sum(d[n+k]*h**k/sp.factorial(k) for k in range(0, 4))
def A(m, Lmax=4):
    tot = 0
    for j in range(m+1):
        for l in range(Lmax+1):
            if 2*l - j < 0: continue
            kl = sp.Rational((-1)**l, 4**l*sp.factorial(l)**2)
            tot += (sp.binomial(m, j)*c**j*s**(m-j)*kl*sp.factorial(2*l)/sp.factorial(2*l-j)
                    *a_der_at_Sc(2*l+m-j)*(-S*s)**(2*l-j))
    return sp.expand(sp.series(tot, th, 0, ORDER).removeO())


def main():
    bnames = {d[0]: sp.Symbol('a')}
    bnames.update({d[n]: sp.Symbol('b' + "'"*(n-1)) for n in range(1, N)})
    res = {}
    for m in range(5):
        e = A(m)
        res[m] = e
        print(f"A_{m}:")
        for k in range(ORDER):
            co = sp.factor_terms(sp.expand(e.coeff(th, k)))
            if co != 0:
                print(f"   th^{k}:", sp.expand(co.xreplace(bnames)))
    # true coefficient g_S = A2 + 1/2 B_S', B_S = A0'
    def dS(e):
        return sp.expand(sp.diff(e, S) + sum(sp.diff(e, d[n])*d[n+1] for n in range(N-1)))
    gS = sp.expand(res[2] + sp.Rational(1, 2)*dS(dS(res[0])))
    print("g_S = A2 + 1/2 B_S' :")
    for k in range(ORDER):
        co = sp.expand(gS.coeff(th, k))
        if co != 0: print(f"   th^{k}:", co.xreplace(bnames))
    T3x12 = sp.expand(res[3] + sp.Rational(3, 4)*dS(dS(res[1])))
    print("12 T3 = A3 + 3/4 A1'' :")
    for k in range(ORDER):
        co = sp.expand(T3x12.coeff(th, k))
        if co != 0: print(f"   th^{k}:", co.xreplace(bnames))
    T2 = gS/4
    T4x48 = sp.expand(res[4] + 4*dS(dS(T2)) - sp.Rational(3, 8)*dS(dS(dS(dS(res[0])))))
    print("48 T4 = A4 + 4 T2'' - 3/8 A0'''' :")
    for k in range(ORDER):
        co = sp.expand(T4x48.coeff(th, k))
        if co != 0: print(f"   th^{k}:", co.xreplace(bnames))


if __name__ == '__main__':
    main()
