from subprocess import check_output
from sage.all import *
from sage.groups.generic import bsgs

def run_kangaroo(target, g, order, bound=None, num_retry=3):
    # solve target = g**k mod p
    if bound is None:
        bound = order
    assert order % 2 == 1
    p = g.parent().characteristic()
    inp = f"{p} {g} {order} {target} {bound}\n"
    print(f"Running kangaroo with {bound.bit_length()} bits...")
    print(inp)
    ret = None
    for _ in range(num_retry):
        try:
            ret = check_output("./kangaroo_gpu", input=inp.encode(), shell=True).decode()
            assert "Solution:" in ret
            break
        except Exception as e:
            print(e)
            continue
    assert ret is not None, "Kangaroo failed"
    print(ret)
    k = ret.split("Solution:")[1]
    k = k.split("\n")[0]
    return ZZ(k)

def dlp_generic(target, g, order, max_bound, num_retry=3):
    if max_bound.bit_length() > 32:
        return run_kangaroo(target, g, order, max_bound, num_retry)
    else:
        return bsgs(g, target, bounds=(ZZ(0), max_bound))

def dlp_prime(target, g, order, num_retry=3):
    if order.bit_length() > 34:
        return run_kangaroo(target, g, order, num_retry=num_retry)
    else:
        os = matrix(ZZ, [[order, 1]])
        return pari(target).znlog(pari(g), pari([order, os])).sage()

def dlp_primepower(target, g, prime, n, num_retry=3):
    order = prime**n
    k = 0
    g0 = g ** (order // prime)
    for i in range(n):
        t0 = target * (g**(-k))
        t0 = t0 ** (prime**(n-1-i))
        assert t0 ** prime == 1
        k += dlp_prime(t0, g0, prime, num_retry) * prime**i
    return k

def dlp_main(target, g, order, max_bound, factors, num_retry=3):
    # solve target = g**k mod p
    cur_bound = max_bound
    mods = []
    for prime, n in factors:
        if cur_bound <= 10:
            break
        if prime > cur_bound: # too large, fall back to generic
            break
        mod = prime**n
        ord_ = order // mod
        g0 = g ** ord_
        t0 = target ** ord_
        cur_bound = cur_bound // mod
        k_ = dlp_primepower(t0, g0, prime, n, num_retry)
        mods.append((k_, mod))
    rs, ms = zip(*mods)
    k0 = crt(list(rs), list(ms))
    MOD = prod(ms) # k = k0 + ? * MOD
    g1 = g ** MOD
    t1 = target * (g ** (-k0))
    k1 = dlp_generic(t1, g1, order // MOD, cur_bound+1, num_retry)
    return k0 + k1 * MOD

def dlp(target, g, order=None, bounds=None, known_factors=None, num_retry=3):
    """
    Solve DLP for target = g**k mod p
    
    - **order**: order of g. A multiple of the order (e.g. p-1) is also accepted.
    - **bounds**: The bound for k. If None, it will be set to `order-1`. 
                 If integer, search range is [0, bound]. 
                 If list/tuple, represents an interval [min, max].
    - **known_factors**: List of known factors of order. The rest part of the order will be handled by generic method.
                        Either (prime, exponent) pairs or prime lists are accepted.
    - **num_retry**: Number of retries when kangaroo fails.
    """
    assert g.parent().is_prime_field()
    p = g.parent().characteristic()
    if order is None:
        order = p-1
    else:
        order = ZZ(order)
    if bounds is None:
        bounds = order
    if known_factors is None:
        known_factors = list(factor(order, algorithm="ecm"))
    if isinstance(known_factors[0], (list, tuple)):
        known_primes = [p for p, _ in known_factors]
    else:
        known_primes = known_factors
    if 2 not in known_primes:
        known_primes.append(2)
    assert g ** order == 1
    known_primes.sort()
    for p in known_primes:
        while (g ** (order // p)) == 1:
            order = order // p
    assert g ** order == 1
    factors = []
    ord_ = order
    for p in known_primes:
        if ord_ % p != 0:
            continue
        v = ord_.valuation(p)
        factors.append((p, v))
        ord_ = ord_ // (p**v)
    if ord_ > 1:
        factors.append((ord_, 1))
    factors.sort()
    if bounds is None:
        bounds = (0, order-1)
    elif isinstance(bounds, (list, tuple)):
        bounds = tuple(bounds)
    else:
        bounds = (0, int(bounds))
    min_bound, max_bound = bounds
    t2 = target * (g ** (-min_bound))
    range = max_bound - min_bound
    return min_bound + dlp_main(t2, g, order, range, factors, num_retry)

if __name__ == "__main__":
    p = ZZ(142444047963504538635221730113223084567757407616109994438418160780448532591183820749754550652125515660094618514124079635895828317494392926414367680859144987898864698703998597459848572924487169639332991108861984848210605189620050761429990054182665913566345731001414332487388962603771226757923933526170354984601)
    known_factors = [2, 3, 5, 11, 127, 168264221, 423525353, 895331897, 1024043681, 1456458758019907, 85060796629905169]
    # 543461853152109886186548651462562402677521158269612949000673937958122534450909090819626646348854329517124504519824819, 1170610126692217622600332376146407931975941165959330923039693472816037457037009467376994100204271791062290277641896913
    g = mod(1337, p) ** 2025
    k = randint(0, 2**300)
    target = g ** k
    k2 = dlp(target, g, order=p-1, bounds=2**300, known_factors=known_factors)
    assert k == k2
