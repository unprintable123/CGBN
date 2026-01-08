#include "ecc.h"


int main()
{
    ECParameters param;
    // y^2 = x^3 + 7
    mpz_set_ui(param.a, 0);
    mpz_set_ui(param.b, 7);
    mpz_set_str(param.p, "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f", 16);

    init_param(param);
    ECPoint G0, P0;
    mpz_set_str(G0.x, "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 16);
    mpz_set_str(G0.y, "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8", 16);
    assert(is_on_curve(G0, param));
    ECPointExtended G, P;
    extend_point(G, G0, param);
    assert(is_on_curve(G, param));
    G0.to_mont_(param);

    for (int i=0; i<1000; i++) {
        point_mul(G0, G0, -405198663214378, param);
        point_mul(G, G, -405198663214378, param);
    }
    proj_point(P0, G, param, true);
    
    assert(P0==G0);
}


