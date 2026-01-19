from subprocess import check_output
from sage.all import *

def run_kangaroo(target, g, p, order, bound=None, num_retry=3):
    # solve target = g**k mod p
    if bound is None:
        bound = order
    assert order % 2 == 1
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
    return int(k, 16)

p = 75509279617903094897049195508623294175933486682745150148271044669446107925169617655137908563221732052272539058456953355591650195253890264002616871262238572973171153673924146708213950106224204992302394414131193429901928741047564864093839754238585137688041378927506813952350305439556094357442133402077693018149
order = 18877319904475773724262298877155823543983371670686287537067761167361526981292404413784477140805433013068134764614238338897912548813472566000654217815559643243292788418481036677053487526556051248075598603532798357475482185261891216023459938559646284422010344731876703488087576359889023589360533350519423254537

g = pow(1337, 2048, p)
bound = 2**62
k = randint(0, bound)
target = pow(g, k, p)

print(run_kangaroo(target, g, p, order, bound))

