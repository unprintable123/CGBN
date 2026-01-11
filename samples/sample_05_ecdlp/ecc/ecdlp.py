from subprocess import check_output
from sage.all import *

def run_kangaroo(Q, P, order=None, bound=None, num_retry=3):
    # solve Q = k*P
    E = P.curve()
    if order is None:
        order = P.order()
    if bound is None:
        bound = order
    p = E.base_ring().characteristic()
    a1, a2, a3, a4, a6 = E.ainvs()
    assert a1 == a2 == a3 == 0
    inp = f"{p} {a4} {a6} {order} {P.x()} {P.y()} {Q.x()} {Q.y()} {bound}\n"
    print(f"Running kangaroo with {bound.bit_length()} bits...")
    print(inp)
    ret = None
    for _ in range(num_retry):
        try:
            ret = check_output(["./kangaroo_gpu"], input=inp.encode()).decode()
            assert "Solution:" in ret
        except Exception as e:
            continue
    assert ret is not None, "Kangaroo failed"
    k = ret.split("Solution:")[1]
    k = k.split("\n")[0]
    return int(k, 16)

def ecdlp_generic(Q, P, order, max_bound, num_retry=3):
    if max_bound.bit_length() > 34:
        return run_kangaroo(Q, P, order, max_bound, num_retry)
    else:
        return discrete_log(Q, P, order, bounds=(0, max_bound), operation="+")

def ecdlp_prime(Q, P, order, num_retry=3):
    if order.bit_length() > 36:
        return run_kangaroo(Q, P, order, num_retry=num_retry)
    else:
        return Q.log(P)

def ecdlp_primepower(Q, P, prime, n, num_retry=3):
    order = prime**n
    k = 0
    P0 = P * (order//prime)
    for i in range(n):
        Q0 = Q - k * P
        Q0 = Q0 * (prime**(n-1-i))
        assert (Q0*prime).is_zero()
        k += ecdlp_prime(Q0, P0, prime, num_retry) * prime**i
    return k

def ecdlp_weierstrass(Q, P, order, max_bound, factors, num_retry=3):
    # solve Q = k*P
    cur_bound = max_bound
    mods = []
    for prime, n in factors:
        if cur_bound <= 10:
            break
        if prime > cur_bound: # too large, fall back to generic
            break
        mod = prime**n
        ord_ = order // mod
        P0 = P * ord_
        Q0 = Q * ord_
        cur_bound = cur_bound // mod
        k_ = ecdlp_primepower(Q0, P0, prime, n, num_retry)
        mods.append((k_, mod))
    rs, ms = zip(*mods)
    k0 = crt(list(rs), list(ms))
    MOD = prod(ms) # k = k0 + ? * MOD
    P1 = P * MOD
    Q1 = Q - k0 * P
    k1 = ecdlp_generic(Q1, P1, order, cur_bound+1, num_retry)
    return k0 + k1 * MOD

def ecdlp(Q, P, order=None, bounds=None, factors=None, num_retry=3):
    """
    Solve ECDLP for Q = k*P over prime field
    
    - **order**: Order of P. A multiple of the order (e.g. the curve order) is also accepted.
    - **bounds**: The bound for k. If None, it will be set to `order-1`. 
                 If integer, search range is [0, bound]. 
                 If list/tuple, represents an interval [min, max].
    - **factors**: List of factors of order.
    - **num_retry**: Number of retries when kangaroo fails.
    """
    if order is None:
        order = P.order()
    if factors is None:
        factors = list(factor(order))
    assert (P*order).is_zero()
    factors_fixed = []
    for p, n in factors:
        while (P*(order//p)).is_zero():
            order = order // p
            n -= 1
        if n > 0:
            factors_fixed.append((p, n))
    factors = factors_fixed
    P.set_order(order)
    assert (Q*order).is_zero()
    E = P.curve()
    assert E.base_ring().is_prime_field(), "Only F_p is supported"
    if not all(a == 0 for a in E.ainvs()[:3]):
        E_short = E.short_weierstrass_model()
        E_short.set_order(E.order())
        phi = E.isomorphism_to(E_short)
        P = phi(P)
        Q = phi(Q)
        E = E_short
        P.set_order(order)
    factors.sort()
    if bounds is None:
        bounds = (0, order-1)
    elif isinstance(bounds, (list, tuple)):
        bounds = tuple(bounds)
    else:
        bounds = (0, int(bounds))
    min_bound, max_bound = bounds
    Q2 = Q - min_bound * P
    range = max_bound - min_bound
    return min_bound + ecdlp_weierstrass(Q2, P, order, range, factors, num_retry)



if __name__ == "__main__":
    # p = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
    p = random_prime(2**256-1, proof=False)
    K = GF(p)
    E = EllipticCurve(K, [randint(0, 65536), randint(0, 65536), randint(0, 65536), randint(0, 65536), randint(0, 65536)])
    order = E.order()
    factors = list(factor(order))
    print(E)
    print(E.order().factor())
    G = E.random_point()
    k = randint(0, 2**64)
    print(k, G)
    wt = walltime()
    k2 = ecdlp(k*G, G, order=order, bounds=2**64, factors=factors)
    print(f'Found in {walltime(wt)}s')
    assert k == k2

