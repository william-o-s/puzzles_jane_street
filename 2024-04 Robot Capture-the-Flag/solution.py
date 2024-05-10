"""
    Solves the April 2024 Robot Capture-the-Flag puzzle using gradient descent.
"""

from sympy import acos, sqrt, pi, diff, nsolve, Symbol

ALPHA = 0.01
MARGIN = 10**-11

E = Symbol('E', positive=True)
s = sqrt(E * (2 - E))
f = ((32 + 3 * pi) * E**2 - 8 * (1 + E) * s + 24 * acos(s)) / (24 * pi)

def gradient(e0):
    '''Returns the gradient of f at e0'''
    return diff(f, E).evalf(subs={E: e0})

e0 = 1
while gradient(e0) > MARGIN:
    e0 = e0 - ALPHA * gradient(e0)

ans = f.subs(E, e0)

print(f"e0          = {e0:.10f}")
print(f"ans = f(e0) = {ans:.10f}")
