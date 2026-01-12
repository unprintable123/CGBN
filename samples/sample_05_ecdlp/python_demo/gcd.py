N = 30

def divsteps_n_matrix(zeta, f, g):
    """Compute zeta and transition matrix t after N divsteps (multiplied by 2^N)."""
    u, v, q, r = 1, 0, 0, 1 # start with identity matrix
    for _ in range(N):
        c1 = zeta >> 63
        # Compute x, y, z as conditionally-negated versions of f, u, v.
        x, y, z = (f ^ c1) - c1, (u ^ c1) - c1, (v ^ c1) - c1
        c2 = -(g & 1)
        # Conditionally add x, y, z to g, q, r.
        g, q, r = g + (x & c2), q + (y & c2), r + (z & c2)
        c1 &= c2                     # reusing c1 here for the earlier c3 variable
        zeta = (zeta ^ c1) - 1       # inlining the unconditional zeta decrement here
        # Conditionally add g, q, r to f, u, v.
        f, u, v = f + (g & c1), u + (q & c1), v + (r & c1)
        # When shifting g down, don't shift q, r, as we construct a transition matrix multiplied
        # by 2^N. Instead, shift f's coefficients u and v up.
        g, u, v = g >> 1, u << 1, v << 1
    return zeta, (u, v, q, r)

def update_fg(f, g, t):
    """Multiply matrix t/2^N with [f, g]."""
    u, v, q, r = t
    cf, cg = u*f + v*g, q*f + r*g
    return cf >> N, cg >> N

def update_de(d, e, t, M, Mi):
    """Multiply matrix t/2^N with [d, e], modulo M."""
    u, v, q, r = t
    d_sign, e_sign = d >> 257, e >> 257
    md, me = (u & d_sign) + (v & e_sign), (q & d_sign) + (r & e_sign)
    cd, ce = (u*d + v*e) % 2**N, (q*d + r*e) % 2**N
    md -= (Mi*cd + md) % 2**N
    me -= (Mi*ce + me) % 2**N
    if (abs(u)+abs(v)+abs(md)) > 2**31:
        print(u, v, md)
    cd, ce = u*d + v*e + M*md, q*d + r*e + M*me
    return cd >> N, ce >> N

def normalize(sign, v, M):
    """Compute sign*v mod M, where v in (-2*M,M); output in [0,M)."""
    assert v < M and v > -2 * M, v
    v_sign = v >> 257
    # Conditionally add M to v.
    v += M & v_sign
    # c = (sign - 1) >> 1
    c = -1 if sign < 0 else 0
    # Conditionally negate v.
    v = (v ^ c) - c
    v_sign = v >> 257
    # Conditionally add M to v again.
    v += M & v_sign
    return v

def modinv(M, Mi, x):
    """Compute the modular inverse of x mod M, given Mi=1/M mod 2^N."""
    zeta, f, g, d, e = -1, M, x, 0, 1
    for _ in range((590 + N - 1) // N):
        zeta, t = divsteps_n_matrix(zeta, f % 2**N, g % 2**N)
        f, g = update_fg(f, g, t)
        d, e = update_de(d, e, t, M, Mi)
    return normalize(f, d, M)

def count_trailing_zeros(v):
    """
    When v is zero, consider all N zero bits as "trailing".
    For a non-zero value v, find z such that v=(d<<z) for some odd d.
    """
    if v == 0:
        return N
    else:
        return (v & -v).bit_length() - 1

NEGINV16 = [15, 5, 3, 9, 7, 13, 11, 1] # NEGINV16[n//2] = (-n)^-1 mod 16, for odd n
def divsteps_n_matrix_var(eta, f, g):
    """Compute eta and transition matrix t after N divsteps (multiplied by 2^N)."""
    u, v, q, r = 1, 0, 0, 1
    i = N
    while True:
        zeros = min(i, count_trailing_zeros(g))
        eta, i = eta - zeros, i - zeros
        g, u, v = g >> zeros, u << zeros, v << zeros
        if i == 0:
            break
        if eta < 0:
            eta, f, u, v, g, q, r = -eta, g, q, r, -f, -u, -v
        limit = min(min(eta + 1, i), 4)
        w = (g * NEGINV16[(f & 15) // 2]) % (2**limit)
        g, q, r = g + w*f, q + w*u, r + w*v
    return eta, (u, v, q, r)

def modinv_var(M, Mi, x):
    """Compute the modular inverse of x mod M, given Mi = 1/M mod 2^N."""
    eta, f, g, d, e = -1, M, x, 0, 1
    iter = 0
    while g != 0:
        iter += 1
        eta, t = divsteps_n_matrix_var(eta, f % 2**N, g % 2**N)
        f, g = update_fg(f, g, t)
        d, e = update_de(d, e, t, M, Mi)
    return normalize(f, d, M), iter

import random
M = 2**255 - 1221621397
Mi = pow(M, -1, 2**N)


cnt = 0
k = 10000
max_iter = 0
for _ in range(k):
    a = random.getrandbits(256)
    a_inv, iter = modinv_var(M, Mi, a)
    cnt += iter
    max_iter = max(max_iter, iter)
    assert a_inv >= 0 and a_inv < M
    assert a_inv * a % M == 1

print(cnt/k, max_iter)










