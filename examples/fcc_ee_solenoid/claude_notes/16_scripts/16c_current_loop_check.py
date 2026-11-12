"""Note 16, step 5: check int B dS along tilted lines against the exact
Biot-Savart field of a current loop and a thick coil (mu0*I = 1).
Expect int B_S = int B_z = 1, int B_X = tan(theta/2), int B_Y = 0, for all offsets.
"""
import numpy as np
from scipy.special import ellipk, ellipe
from scipy.integrate import quad
# exact field of a circular loop, radius R at z=z0, mu0*I = 1  ->  int B.dl through loop = 1
def loopB(x, y, z, R=1.0, z0=0.0):
    r = np.hypot(x, y); zz = z - z0
    a2 = (R - r)**2 + zz**2
    b2 = (R + r)**2 + zz**2
    k2 = 4*R*r/b2
    K, E = ellipk(k2), ellipe(k2)
    pref = 1/(2*np.pi*np.sqrt(b2))
    Bz = pref*(K + (R**2 - r**2 - zz**2)/a2*E)
    if r < 1e-12:
        Br = 0.0
    else:
        Br = pref*zz/r*(-K + (R**2 + r**2 + zz**2)/a2*E)
    return Br*x/max(r,1e-300), Br*y/max(r,1e-300), Bz
def coil(x, y, z):  # a short thick coil: loops of radius 0.8..1.2, z0 in [-0.6,0.6]
    Bx=By=Bz=0.0; n=0
    for R in (0.8, 1.0, 1.2):
        for z0 in np.linspace(-0.6, 0.6, 5):
            bx,by,bz = loopB(x,y,z,R,z0); Bx+=bx; By+=by; Bz+=bz; n+=1
    return Bx/n, By/n, Bz/n
def integ(th, X0, Y0, field, Smax=400):
    c, s = np.cos(th), np.sin(th)
    def pt(S):  # beam axis offset by X0,Y0 ; x = X c - S s ; z = S c + X s
        return X0*c - S*s, Y0, S*c + X0*s
    out = []
    for comp in range(4):
        def f(S):
            bx, by, bz = field(*pt(S))
            BX = c*bx + s*bz; BS = -s*bx + c*bz
            return (BS, BX, by, bz)[comp]
        v = sum(quad(f, a, b, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
                for a, b in [(-Smax, -5), (-5, 5), (5, Smax)])
        out.append(v)
    return out
for name, field in [("single loop", loopB), ("thick coil", coil)]:
    print(name)
    for th in (0.015, 0.2, 0.5):
        for (X0, Y0) in [(0, 0), (0.1, 0), (0.2, 0), (0, 0.1), (0.1, 0.1)]:
            BS, BX, BY, Bz = integ(th, X0, Y0, field)
            print(f"  th={th:5.3f} X0={X0:3.1f} Y0={Y0:3.1f}: intBS={BS:.8f} intBz={Bz:.8f} "
                  f"intBX/tan(th/2)={BX/np.tan(th/2):.8f} intBY={BY:.2e}")
