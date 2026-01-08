#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <gmp.h>
#include <chrono>
#include <assert.h>

constexpr uint32_t TPI = 4;
constexpr uint32_t NBITS = 256;
constexpr uint32_t NLIMBS = (NBITS+32*(TPI-1))/(TPI*32);
constexpr uint32_t RBITS = NLIMBS * (TPI*32);

struct ECParameters
{
    mpz_t a, b, p;  // y^2 = x^3 + ax + b (mod p)
    mpz_t _r, _r2, _r3, _p_inv_neg; // R mod p, R^2 mod p, R^3 mod p, -p^-1 mod R
    mpz_t _a_mont, _b_mont; // a, b in montgomery space

    ECParameters() {
        mpz_inits(a, b, p, _r, _r2, _r3, _p_inv_neg, _a_mont, _b_mont, NULL);
    }

    ~ECParameters() {
        mpz_clears(a, b, p, _r, _r2, _r3, _p_inv_neg, _a_mont, _b_mont, NULL);
    }
};

struct ECPoint
{
    mpz_t x, y;
    bool z; // 0 is infinity
    bool _is_mont=false;

    ECPoint(bool infinity = false) {
        mpz_init(x);
        mpz_init(y);
        z = (~infinity);
    }

    ~ECPoint() {
        mpz_clear(x);
        mpz_clear(y);
    }

    void to_mont_(ECParameters &param);
    void from_mont_(ECParameters &param);

    bool operator==(ECPoint &p) {
        if (z == 0) {
            return p.z == 0;
        } else {
            return p.z == 1 && mpz_cmp(x, p.x) == 0 && mpz_cmp(y, p.y) == 0 && _is_mont == p._is_mont;
        }
    }
};

struct ECPointExtended {
    mpz_t x, y, z; // in montgomery space

    ECPointExtended() {
        mpz_init(x);
        mpz_init(y);
        mpz_init(z);
    }

    ~ECPointExtended() {
        mpz_clear(x);
        mpz_clear(y);
        mpz_clear(z);
    }
};

#define THREAD_LOCAL_MPZ_WORKSPACE(...) \
    thread_local static bool _ws_init = false; \
    thread_local static mpz_t __VA_ARGS__; \
    if (!_ws_init) { \
        mpz_inits(__VA_ARGS__, NULL); \
        _ws_init = true; \
    }

bool is_infinity(ECPointExtended &p) {
    return mpz_sgn(p.z) == 0;
}

bool is_infinity(ECPoint &p) {
    return p.z == 0;
}

void print_mpz(mpz_t &value) {
    mpz_out_str(stdout, 16, value);
    printf("\n");
}

void mpz_randombits(mpz_t &value, uint32_t bits, bool sign=false) {
    thread_local static bool _ws_init = false;
    thread_local static gmp_randstate_t _state;
    if (!_ws_init) {
        gmp_randinit_mt(_state);
        unsigned long int seed;
        FILE *fp = fopen("/dev/urandom", "rb");
        assert(fp != NULL);
        auto ret = fread(&seed, sizeof(unsigned long int), 1, fp);
        assert(ret == 1);
        fclose(fp);
        gmp_randseed_ui(_state, seed);
        srand(time(NULL));
        _ws_init = true;
    }

    mpz_urandomb(value, _state, bits);
    if (sign && (rand() & 1)) {
        mpz_neg(value, value);
    }
}

void mont_add(mpz_t &r, mpz_t &a, mpz_t &b, ECParameters &param) {
    mpz_add(r, a, b);
    if (mpz_cmp(r, param.p) >= 0) {
        mpz_sub(r, r, param.p);
    }
}

void mont_sub(mpz_t &r, mpz_t &a, mpz_t &b, ECParameters &param) {
    mpz_sub(r, a, b);
    if (mpz_sgn(r) < 0) {
        mpz_add(r, r, param.p);
    }
}

void mont_redc(mpz_t &r, mpz_t &a, ECParameters &param) {
    THREAD_LOCAL_MPZ_WORKSPACE(tmp);

    mpz_tdiv_r_2exp(tmp, a, RBITS);
    mpz_mul(tmp, tmp, param._p_inv_neg);
    mpz_tdiv_r_2exp(tmp, tmp, RBITS);
    mpz_mul(tmp, tmp, param.p);
    mpz_add(tmp, tmp, a);
    mpz_tdiv_q_2exp(r, tmp, RBITS);

    if (mpz_cmp(r, param.p) >= 0)
    {
        mpz_sub(r, r, param.p);
    }
}

void mont_mul(mpz_t &r, mpz_t &a, mpz_t &b, ECParameters &param) {
    mpz_mul(r, a, b);
    mont_redc(r, r, param);
}

void mont_sqr(mpz_t &r, mpz_t &a, ECParameters &param) {
    mpz_mul(r, a, a);
    mont_redc(r, r, param);
}

void mont_inv(mpz_t &r, mpz_t &a, ECParameters &param) {
    mpz_invert(r, a, param.p);
    mont_mul(r, r, param._r3, param);
}

void to_mont(mpz_t &r, mpz_t &x, ECParameters &param) {
    mont_mul(r, x, param._r2, param);
}

void from_mont(mpz_t &r, mpz_t &x, ECParameters &param) {
    mont_redc(r, x, param);
}

void ECPoint::to_mont_(ECParameters &param) {
    assert(!_is_mont);
    to_mont(x, x, param);
    to_mont(y, y, param);
    _is_mont = true;
}

void ECPoint::from_mont_(ECParameters &param) {
    assert(_is_mont);
    from_mont(x, x, param);
    from_mont(y, y, param);
    _is_mont = false;
}

void init_param(ECParameters &param) {
    size_t bits = mpz_sizeinbase(param.p, 2);
    assert(bits <= NBITS);

    mpz_t R;
    mpz_init(R);
    mpz_set_ui(R, 1);
    mpz_mul_2exp(R, R, RBITS);

    mpz_mod(param._r, R, param.p);

    mpz_mul(param._r2, R, R);
    mpz_mod(param._r2, param._r2, param.p);

    mpz_invert(param._p_inv_neg, param.p, R);
    mpz_sub(param._p_inv_neg, R, param._p_inv_neg);

    mont_sqr(param._r3, param._r2, param);

    to_mont(param._a_mont, param.a, param);
    to_mont(param._b_mont, param.b, param);
    mpz_clear(R);
}

void extend_point(ECPointExtended &p1, ECPoint &p0, ECParameters &param) {
    if (p0.z == 0) {
        mpz_set_ui(p1.x, 0);
        mpz_set_ui(p1.y, 0);
        mpz_set_ui(p1.z, 0);
    } else {
        if (p0._is_mont) {
            mpz_set(p1.x, p0.x);
            mpz_set(p1.y, p0.y);
        } else {
            to_mont(p1.x, p0.x, param);
            to_mont(p1.y, p0.y, param);
        }
        mpz_set(p1.z, param._r);
    }
}

void proj_point(ECPoint &p0, ECPointExtended &p1, ECParameters &param, bool is_mont=false) {
    p0._is_mont = is_mont;
    if (is_infinity(p1)) {
        p0.z = 0;
    } else {
        p0.z = 1;
        THREAD_LOCAL_MPZ_WORKSPACE(z_inv);

        mpz_invert(z_inv, p1.z, param.p);
        if (is_mont) {
            mont_mul(z_inv, z_inv, param._r3, param);
        } else {
            mont_mul(z_inv, z_inv, param._r2, param);
        }
        mont_mul(p0.x, p1.x, z_inv, param);
        mont_mul(p0.y, p1.y, z_inv, param);
    }
}

void print_point(ECPoint &p) {
    printf("Point: (0x");
    mpz_out_str(stdout, 16, p.x);
    printf(", 0x");
    mpz_out_str(stdout, 16, p.y);
    printf(") mont: ");
    if (p._is_mont) {
        printf("true\n");
    } else {
        printf("false\n");

    }
}

void print_point(ECPointExtended &p) {
    printf("Projective Point: (0x");
    mpz_out_str(stdout, 16, p.x);
    printf(", 0x");
    mpz_out_str(stdout, 16, p.y);
    printf(", 0x");
    mpz_out_str(stdout, 16, p.z);
    printf(")\n");
}


bool is_on_curve(ECPointExtended &p, ECParameters &param) {
    // y^2 * z = x^3 + a * x * z^2 + b * z^3
    if (is_infinity(p))
    {
        return true;
    }

    mpz_t t1, t2;
    mpz_init(t1);
    mpz_init(t2);

    
    mont_mul(t1, p.x, param._a_mont, param);
    mont_mul(t2, p.z, param._b_mont, param);
    mont_add(t1, t1, t2, param);
    mont_sqr(t2, p.z, param);
    mont_mul(t1, t1, t2, param); // t1 = a * x * z^2 + b * z^3

    mont_sqr(t2, p.y, param);
    mont_mul(t2, t2, p.z, param); // t2 = y^2 * z
    mont_sub(t1, t1, t2, param); // t1 = 

    mont_sqr(t2, p.x, param);
    mont_mul(t2, t2, p.x, param); // t2 = x^3
    mont_add(t1, t1, t2, param); // t1 = x^3 + a * x * z^2 + b * z^3 - y^2 * z

    bool ret = mpz_sgn(t1) == 0;

    mpz_clear(t1);
    mpz_clear(t2);

    return ret;
}

bool is_on_curve(ECPoint &p, ECParameters &param) {
    // y^2 = x^3 + ax + b
    if (is_infinity(p))
    {
        return true;
    }

    mpz_t t1, t2, x_mont, y_mont;
    mpz_init(t1);
    mpz_init(t2);
    mpz_init(x_mont);
    mpz_init(y_mont);

    if (!p._is_mont) {
        to_mont(x_mont, p.x, param);
        to_mont(y_mont, p.y, param);
    } else {
        mpz_set(x_mont, p.x);
        mpz_set(y_mont, p.y);
    }
    mont_sqr(t1, x_mont, param);
    mont_add(t1, t1, param._a_mont, param);
    mont_mul(t1, t1, x_mont, param);
    mont_add(t1, t1, param._b_mont, param); // t1 = x^3 + ax + b
    mont_sqr(t2, y_mont, param); // t2 = y^2

    bool ret = mpz_cmp(t1, t2) == 0;

    mpz_clear(t1);
    mpz_clear(t2);
    mpz_clear(x_mont);
    mpz_clear(y_mont);
    return ret;
}

void point_set(ECPointExtended &r, ECPointExtended &p) {
    mpz_set(r.x, p.x);
    mpz_set(r.y, p.y);
    mpz_set(r.z, p.z);
}

void point_set(ECPoint &r, ECPoint &p) {
    mpz_set(r.x, p.x);
    mpz_set(r.y, p.y);
    r.z = p.z;
    r._is_mont = p._is_mont;
}

void point_neg_(ECPointExtended &r, ECParameters &param) {
    if (mpz_sgn(r.y) == 1)
        mpz_sub(r.y, param.p, r.y);
    else
        assert(mpz_sgn(r.y) == 0);
}

void point_neg_(ECPoint &r, ECParameters &param) {
    if (mpz_sgn(r.y) == 1)
        mpz_sub(r.y, param.p, r.y);
    else
        assert(mpz_sgn(r.y) == 0);
}

void point_add(ECPointExtended &r, ECPointExtended &p1, ECPointExtended &p2, ECParameters &param) { 
    if (is_infinity(p1)) {
        point_set(r, p2);
    } else if (is_infinity(p2)) {
        point_set(r, p1);
    } else {
        THREAD_LOCAL_MPZ_WORKSPACE(x1z2, y1z2, z1z2, u, uu, v, vv, vvv, R, A);

        mont_mul(x1z2, p1.x, p2.z, param);
        mont_mul(y1z2, p1.y, p2.z, param);
        mont_mul(z1z2, p1.z, p2.z, param);

        mont_mul(u, p2.y, p1.z, param);
        mont_sub(u, u, y1z2, param);
        mont_sqr(uu, u, param);

        mont_mul(v, p2.x, p1.z, param);
        mont_sub(v, v, x1z2, param);
        mont_sqr(vv, v, param);
        mont_mul(vvv, vv, v, param);

        assert(mpz_sgn(u) != 0 || mpz_sgn(v) != 0);

        mont_mul(R, vv, x1z2, param);
        mont_mul(A, uu, z1z2, param);
        mont_sub(A, A, vvv, param);
        mont_sub(A, A, R, param);
        mont_sub(A, A, R, param);
        mont_sub(R, R, A, param);

        mont_mul(r.x, v, A, param);
        mont_mul(r.z, vvv, z1z2, param);
        mont_mul(vvv, vvv, y1z2, param);
        mont_mul(r.y, u, R, param);
        mont_sub(r.y, r.y, vvv, param);
    }
}

void point_double(ECPointExtended &r, ECPointExtended &p, ECParameters &param) { 
    if (is_infinity(p)) {
        mpz_set_ui(r.z, 0);
    } else { 
        THREAD_LOCAL_MPZ_WORKSPACE(xx, zz, w, s, ss, sss, R, RR, B, h);

        mont_sqr(xx, p.x, param);
        mont_sqr(zz, p.z, param);
        mont_mul(w, param._a_mont, zz, param);
        mont_add(w, w, xx, param);
        mont_add(w, w, xx, param);
        mont_add(w, w, xx, param); // w = a * zz + 3 * xx
        mont_mul(s, p.y, p.z, param);
        mont_add(s, s, s, param); // s = 2 * y * z
        mont_mul(ss, s, s, param);
        mont_mul(sss, s, ss, param);

        mont_mul(R, p.y, s, param);
        mont_mul(RR, R, R, param);

        mont_add(B, p.x, R, param);
        mont_sqr(B, B, param);
        mont_sub(B, B, RR, param);
        mont_sub(B, B, xx, param);

        mont_sqr(h, w, param);
        mont_sub(h, h, B, param);
        mont_sub(h, h, B, param);

        mont_mul(r.x, h, s, param);
        mont_sub(B, B, h, param);
        mont_mul(r.y, w, B, param);
        mont_sub(r.y, r.y, RR, param);
        mont_sub(r.y, r.y, RR, param);
        mpz_set(r.z, sss);
    }
}

void point_double(ECPoint &r, ECPoint &p, ECParameters &param) { 
    if (is_infinity(p)) {
        point_set(r, p);
    } else {
        if (mpz_cmp_ui(p.y, 0) == 0) {
            r.z = 0;
            return;
        }
        
        THREAD_LOCAL_MPZ_WORKSPACE(lambda, t1, t2, x1_mont_, y1_mont_);

        mpz_t& x1_mont = (!p._is_mont) ? (to_mont(x1_mont_, p.x, param), x1_mont_) : p.x;
        mpz_t& y1_mont = (!p._is_mont) ? (to_mont(y1_mont_, p.y, param), y1_mont_) : p.y;
        mont_sqr(t1, x1_mont, param);
        mont_add(lambda, t1, t1, param);
        mont_add(lambda, lambda, t1, param);
        mont_add(lambda, lambda, param._a_mont, param);
        
        mont_add(t1, y1_mont, y1_mont, param);
        mont_inv(t1, t1, param);
        mont_mul(lambda, lambda, t1, param); // lambda = (3 * x1^2 + a) / (2 * y1)
        
        mont_mul(t1, lambda, lambda, param);
        mont_sub(t1, t1, x1_mont, param);
        mont_sub(t1, t1, x1_mont, param); // x3 = lambda^2 - 2 * x1
        mont_sub(t2, x1_mont, t1, param);
        mpz_set(r.x, t1);
        mont_mul(t1, lambda, t2, param);
        mont_sub(r.y, t1, y1_mont, param); // y3 = lambda * (x1 - x3) - y1

        if (!p._is_mont) {
            from_mont(r.x, r.x, param);
            from_mont(r.y, r.y, param);
        }
        r.z = 1;
        r._is_mont = p._is_mont;
    }
}

void point_add(ECPoint &r, ECPoint &p1, ECPoint &p2, ECParameters &param) { 
    if (is_infinity(p1)) {
        point_set(r, p2);
    } else if (is_infinity(p2)) {
        point_set(r, p1);
    } else {
        if (mpz_cmp(p1.x, p2.x) == 0) {
            if (mpz_cmp(p1.y, p2.y) == 0) {
                point_double(r, p1, param);
            } else {
                mpz_set_ui(r.x, 0);
                mpz_set_ui(r.y, 0);
                r.z = 0;
            }
            return;
        }
        THREAD_LOCAL_MPZ_WORKSPACE(lambda, t1, t2, x1_mont_, x2_mont_, y1_mont_, y2_mont_);
        mpz_t& x1_mont = (!p1._is_mont) ? (to_mont(x1_mont_, p1.x, param), x1_mont_) : p1.x;
        mpz_t& x2_mont = (!p2._is_mont) ? (to_mont(x2_mont_, p2.x, param), x2_mont_) : p2.x;
        mpz_t& y1_mont = (!p1._is_mont) ? (to_mont(y1_mont_, p1.y, param), y1_mont_) : p1.y;
        mpz_t& y2_mont = (!p2._is_mont) ? (to_mont(y2_mont_, p2.y, param), y2_mont_) : p2.y;

        mont_sub(t1, y2_mont, y1_mont, param);
        mont_sub(t2, x2_mont, x1_mont, param);
        mont_inv(t2, t2, param);
        mont_mul(lambda, t1, t2, param); // lambda = (y2 - y1) / (x2 - x1)
        mont_mul(t1, lambda, lambda, param);
        mont_sub(t1, t1, x1_mont, param);
        mont_sub(t1, t1, x2_mont, param); // x3 = lambda^2 - x1 - x2
        mont_sub(t2, x1_mont, t1, param);
        mpz_set(r.x, t1);
        mont_mul(t1, lambda, t2, param);
        mont_sub(r.y, t1, y1_mont, param); // y3 = lambda * (x1 - x3) - y1

        if (!p1._is_mont) {
            from_mont(r.x, r.x, param);
            from_mont(r.y, r.y, param);
        }
        r.z = 1;
        r._is_mont = p1._is_mont;
    }
}

void point_mul(ECPointExtended &r, ECPointExtended &p, mpz_t &n, ECParameters &param) {
    THREAD_LOCAL_MPZ_WORKSPACE(n1);
    if (mpz_sgn(n) == 0 || is_infinity(p)) {
        mpz_set_ui(r.z, 0); 
        return;
    }

    ECPointExtended R0, R1;

    mpz_set(n1, n);
    mpz_set_ui(R0.z, 0);
    point_set(R1, p);
    if (mpz_sgn(n1) < 0) {
        mpz_neg(n1, n1);
        point_neg_(R1, param);
    }

    size_t n_bits = mpz_sizeinbase(n1, 2);

    for (int i = (int)n_bits - 1; i >= 0; i--) {
        if (mpz_tstbit(n1, i)) {
            point_add(R0, R0, R1, param);
            point_double(R1, R1, param);
        } else {
            point_add(R1, R0, R1, param);
            point_double(R0, R0, param);
        }
    }

    mpz_set(r.x, R0.x);
    mpz_set(r.y, R0.y);
    mpz_set(r.z, R0.z);
}


void point_mul(ECPoint &r, ECPoint &p, mpz_t &n, ECParameters &param) {
    ECPointExtended P;
    extend_point(P, p, param);
    point_mul(P, P, n, param);
    proj_point(r, P, param, p._is_mont);
}

template<typename T>
void point_mul(T &r, T &p, signed long int n, ECParameters &param) {
    THREAD_LOCAL_MPZ_WORKSPACE(n_mpz);
    mpz_set_si(n_mpz, n);
    point_mul(r, p, n_mpz, param);
}



