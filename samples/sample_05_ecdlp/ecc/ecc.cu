#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <chrono>
#include "cgbn/cgbn.h"
#include "cgbn/arith/arith.h"
#include "../../utility/support.h"

#include "ecc.h"
#include "modinv.cu"

#define GRP_INV_SIZE 128
#define ADD_CHECK


struct Curve {
    cgbn_mem_t<NBITS> p;
    cgbn_mem_t<NBITS> a;
    cgbn_mem_t<NBITS> b;
    cgbn_mem_t<NBITS> _r;
    cgbn_mem_t<NBITS> _r2;
    cgbn_mem_t<NBITS> _r3;
    uint32_t np0;

    Curve(ECParameters &param) {
      from_mpz(param.p, p._limbs, NBITS/32);
      from_mpz(param._a_mont, a._limbs, NBITS/32);
      from_mpz(param._b_mont, b._limbs, NBITS/32);
      from_mpz(param._r, _r._limbs, NBITS/32);
      from_mpz(param._r2, _r2._limbs, NBITS/32);
      from_mpz(param._r3, _r3._limbs, NBITS/32);
      np0 = mpz_get_ui(param._p_inv_neg);
    }
};

struct ECPointCGBN {
    cgbn_mem_t<NBITS> x;
    cgbn_mem_t<NBITS> y;
    bool z;

    ECPointCGBN(bool infinity = false) {
        z = ~infinity;
    }

    void from_point(ECPoint &point) {
      assert(point._is_mont);
      from_mpz(point.x, x._limbs, NBITS/32);
      from_mpz(point.y, y._limbs, NBITS/32);
      z=point.z;
    }

    void to_point(ECPoint &point) {
      to_mpz(point.x, x._limbs, NBITS/32);
      to_mpz(point.y, y._limbs, NBITS/32);
      point.z=z;
      point._is_mont=true;
    }
};

typedef struct {
  ECPointCGBN A;
  ECPointCGBN B;
  ECPointCGBN C;
} instance_t;


typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, NBITS> env_t;

typedef struct {
  env_t::cgbn_t x;
  env_t::cgbn_t y;
  bool z;
} ECPointGPU;

typedef struct {
    env_t::cgbn_t p;
    env_t::cgbn_t a;
    env_t::cgbn_t b;
    env_t::cgbn_t _r;
    env_t::cgbn_t _r2;
    env_t::cgbn_t _r3;
    uint32_t np0;
} CurveGPU;

struct ECPointExtGPU {
  env_t::cgbn_t x;
  env_t::cgbn_t y;
  env_t::cgbn_t z;

  __device__ ECPointExtGPU() {}

  __device__ ECPointExtGPU(env_t env, ECPointGPU &P, CurveGPU &curve) {
    cgbn_set(env, x, P.x);
    cgbn_set(env, y, P.y);
    cgbn_set(env, z, curve._r);
  }
};

__device__ __forceinline__ void load_point(env_t env, ECPointGPU &r, ECPointCGBN *const address) {
  cgbn_load(env, r.x, &(address->x));
  cgbn_load(env, r.y, &(address->y));
  r.z=address->z;
}

__device__ __forceinline__ void save_point(env_t env, ECPointCGBN *const address, const ECPointGPU &r) {
  cgbn_store(env, &(address->x), r.x);
  cgbn_store(env, &(address->y), r.y);
  address->z = r.z;
}

__device__ __forceinline__ void point_set(env_t env, ECPointGPU &r, const ECPointGPU &P) {
  cgbn_set(env, r.x, P.x);
  cgbn_set(env, r.y, P.y);
  r.z=P.z;
}

template<class env_t>
__device__ __forceinline__ void mont_add(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b, const CurveGPU &curve) {
  auto carry = cgbn_add(env, r, a, b);
  if (carry || cgbn_compare(env, r, curve.p) >= 0)
  {
    cgbn_sub(env, r, r, curve.p);
  }
}

template<class env_t>
__device__ __forceinline__ void mont_sub(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b, const CurveGPU &curve) {
  auto borrow = cgbn_sub(env, r, a, b);
  if (borrow)
  {
    cgbn_add(env, r, r, curve.p);
  }
}

template<class env_t>
__device__ __forceinline__ void mont_mul(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &b, const CurveGPU &curve) {
  cgbn_mont_mul(env, r, a, b, curve.p, curve.np0);
  if (cgbn_compare(env, r, curve.p) >= 0) {
    cgbn_sub(env, r, r, curve.p);
  }
}

template<class env_t>
__device__ __forceinline__ void mont_sqr(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const CurveGPU &curve) {
  cgbn_mont_sqr(env, r, a, curve.p, curve.np0);
  if (cgbn_compare(env, r, curve.p) >= 0) {
    cgbn_sub(env, r, r, curve.p);
  }
}

template<class env_t>
__device__ __forceinline__ void mont_inv(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const CurveGPU &curve) {
  // cgbn_modular_inverse(env, r, a, curve.p);
  yang_modinv_odd(env, r, a, curve.p);
  cgbn_mont_mul(env, r, r, curve._r3, curve.p, curve.np0);
}

__device__ __forceinline__ void point_add(env_t env, ECPointExtGPU &r, const ECPointExtGPU &p1, const ECPointExtGPU &p2, const CurveGPU &curve) { 
  env_t::cgbn_t x1z2, y1z2, z1z2, u, uu, v, vv, vvv, R, A;

  mont_mul(env, x1z2, p1.x, p2.z, curve);
  mont_mul(env, y1z2, p1.y, p2.z, curve);
  mont_mul(env, z1z2, p1.z, p2.z, curve);

  mont_mul(env, u, p2.y, p1.z, curve);
  mont_sub(env, u, u, y1z2, curve);
  mont_sqr(env, uu, u, curve);

  mont_mul(env, v, p2.x, p1.z, curve);
  mont_sub(env, v, v, x1z2, curve);
  mont_sqr(env, vv, v, curve);
  mont_mul(env, vvv, vv, v, curve);

#ifdef ADD_CHECK
  assert(cgbn_equals_ui32(env, u, 0) == 0 || cgbn_equals_ui32(env, v, 0) == 0);
#endif

  mont_mul(env, R, vv, x1z2, curve);
  mont_mul(env, A, uu, z1z2, curve);
  mont_sub(env, A, A, vvv, curve);
  mont_sub(env, A, A, R, curve);
  mont_sub(env, A, A, R, curve);
  mont_sub(env, R, R, A, curve);

  mont_mul(env, r.x, v, A, curve);
  mont_mul(env, r.z, vvv, z1z2, curve);
  mont_mul(env, vvv, vvv, y1z2, curve);
  mont_mul(env, r.y, u, R, curve);
  mont_sub(env, r.y, r.y, vvv, curve);
}

__device__ __forceinline__ void point_double(env_t env, ECPointExtGPU &r, const ECPointExtGPU &P, const CurveGPU &curve) {
  env_t::cgbn_t xx, zz, w, s, ss, sss, R, RR, B, h;

  mont_sqr(env, xx, P.x, curve);
  mont_sqr(env, zz, P.z, curve);
  mont_mul(env, w, curve.a, zz, curve);
  mont_add(env, w, w, xx, curve);
  mont_add(env, w, w, xx, curve);
  mont_add(env, w, w, xx, curve); // w = a * zz + 3 * xx
  mont_mul(env, s, P.y, P.z, curve);
  mont_add(env, s, s, s, curve); // s = 2 * y * z
  mont_mul(env, ss, s, s, curve);
  mont_mul(env, sss, s, ss, curve);

  mont_mul(env, R, P.y, s, curve);
  mont_mul(env, RR, R, R, curve);

  mont_add(env, B, P.x, R, curve);
  mont_sqr(env, B, B, curve);
  mont_sub(env, B, B, RR, curve);
  mont_sub(env, B, B, xx, curve);

  mont_sqr(env, h, w, curve);
  mont_sub(env, h, h, B, curve);
  mont_sub(env, h, h, B, curve);

  mont_mul(env, r.x, h, s, curve);
  mont_sub(env, B, B, h, curve);
  mont_mul(env, r.y, w, B, curve);
  mont_sub(env, r.y, r.y, RR, curve);
  mont_sub(env, r.y, r.y, RR, curve);

  cgbn_set(env, r.z, sss);
}

__device__ __forceinline__ void point_set(env_t env, ECPointExtGPU &r, const ECPointExtGPU &P) {
  cgbn_set(env, r.x, P.x);
  cgbn_set(env, r.y, P.y);
  cgbn_set(env, r.z, P.z);
}

__device__ __forceinline__ void point_mul(env_t env, ECPointExtGPU &r, const ECPointExtGPU &P, const env_t::cgbn_t &n, const CurveGPU &curve) {
  env_t::cgbn_t n_copy;
  ECPointExtGPU A, B;
  cgbn_set(env, n_copy, n);
  point_set(env, A, P);
  point_double(env, B, A, curve);
  bool is_odd = cgbn_get_ui32(env, n_copy) & 1;
  cgbn_shift_right(env, n_copy, n_copy, 1);
  while (cgbn_equals_ui32(env, n_copy, 0)==0) {
    if (cgbn_get_ui32(env, n_copy) & 1) {
      point_add(env, A, A, B, curve);
    }
    point_double(env, B, B, curve);
    cgbn_shift_right(env, n_copy, n_copy, 1);
  }
  if (!is_odd) {
    cgbn_set(env, B.x, P.x);
    cgbn_set(env, B.y, P.y);
    cgbn_set(env, B.z, P.z);
    cgbn_sub(env, B.y, curve.p, B.y);
    point_add(env, A, A, B, curve);
  }
  point_set(env, r, A);
}


__device__ __forceinline__ void proj_point(env_t env, ECPointGPU &r, const ECPointExtGPU &P, const CurveGPU &curve) {
  env_t::cgbn_t z_inv;
  mont_inv(env, z_inv, P.z, curve);
  mont_mul(env, r.x, P.x, z_inv, curve);
  mont_mul(env, r.y, P.y, z_inv, curve);
  r.z=(cgbn_equals_ui32(env, P.z, 0)==0);
}


__device__ __forceinline__ void point_add(env_t env, ECPointGPU &r, const ECPointGPU &P, const ECPointGPU &Q, const CurveGPU &curve) {
#ifdef ADD_CHECK
  if (P.z==0) {
    point_set(env, r, Q);
    return;
  } else if (Q.z==0) {
    point_set(env, r, P);
    return;
  }
  assert(cgbn_compare(env, P.x, Q.x)!=0);
#endif
  env_t::cgbn_t lambda, t1, t2;
  mont_sub(env, t1, P.y, Q.y, curve);
  mont_sub(env, t2, P.x, Q.x, curve);
  mont_inv(env, lambda, t2, curve);
  mont_mul(env, lambda, lambda, t1, curve); // lambda = (y2-y1)/(x2-x1)
  mont_sqr(env, t1, lambda, curve);
  mont_sub(env, t1, t1, P.x, curve);
  mont_sub(env, t1, t1, Q.x, curve); // x3 = lambda^2 - x1 - x2
  mont_sub(env, t2, P.x, t1, curve);
  cgbn_set(env, r.x, t1);
  mont_mul(env, t1, lambda, t2, curve);
  mont_sub(env, r.y, t1, P.y, curve); // y3 = lambda*(x1-x3) - y1
  r.z=1;
}

__device__ __forceinline__ void point_double(env_t env, ECPointGPU &r, const ECPointGPU &P, const CurveGPU &curve) {
#ifdef ADD_CHECK
  if (P.z==0) {
    r.z=0;
    return;
  }
#endif
  env_t::cgbn_t lambda, t1, t2;
  mont_sqr(env, t1, P.x, curve);
  mont_add(env, lambda, t1, t1, curve);
  mont_add(env, lambda, lambda, t1, curve);
  mont_add(env, lambda, lambda, curve.a, curve);

  mont_add(env, t1, P.y, P.y, curve);
  mont_inv(env, t1, t1, curve);
  mont_mul(env, lambda, lambda, t1, curve); // lambda = (3*x1^2+a)/(2*y1)

  mont_sqr(env, t1, lambda, curve);
  mont_sub(env, t1, t1, P.x, curve);
  mont_sub(env, t1, t1, P.x, curve); // x3 = lambda^2 - 2*x1
  mont_sub(env, t2, P.x, t1, curve);
  cgbn_set(env, r.x, t1);
  mont_mul(env, t1, lambda, t2, curve);
  mont_sub(env, r.y, t1, P.y, curve); // y3 = lambda*(x1-x3) - y1
}

__device__ __forceinline__ void batch_inverse(env_t env, env_t::cgbn_t *out, env_t::cgbn_t *inv, CurveGPU &curve) {
  // compute out[i]*(inv[i]^-1)
  env_t::cgbn_t prod;
  cgbn_set(env, prod, curve._r);
  for (int i=0; i<GRP_INV_SIZE; i++) {
    mont_mul(env, out[i], out[i], prod, curve);
    mont_mul(env, prod, prod, inv[i], curve);
  }
  mont_inv(env, prod, prod, curve);
  for (int i=GRP_INV_SIZE-1; i>=0; i--) {
    mont_mul(env, out[i], out[i], prod, curve);
    mont_mul(env, prod, prod, inv[i], curve);
  }
}

__device__ __forceinline__ void batch_point_add(env_t env, ECPointGPU *Rs, ECPointGPU *Ps, ECPointGPU *Qs, CurveGPU &curve) { 
#ifdef ADD_CHECK
for (int i=0; i<GRP_INV_SIZE; i++) {
  auto &P = Ps[i];
  auto &Q = Qs[i];
  assert(P.z!=0);
  assert(Q.z!=0);
  assert(cgbn_compare(env, P.x, Q.x)!=0);
}
#endif
  env_t::cgbn_t dy[GRP_INV_SIZE], dx[GRP_INV_SIZE];
  for (int i=0; i<GRP_INV_SIZE; i++) {
    auto &P = Ps[i];
    auto &Q = Qs[i];
    mont_sub(env, dy[i], P.y, Q.y, curve);
    mont_sub(env, dx[i], P.x, Q.x, curve);
  }
  batch_inverse(env, dy, dx, curve);
  env_t::cgbn_t t1, t2;
  for (int i=0; i<GRP_INV_SIZE; i++) {
    auto &P = Ps[i];
    auto &Q = Qs[i];
    auto &R = Rs[i];
    auto &lambda = dy[i];
    mont_sqr(env, t1, lambda, curve);
    mont_sub(env, t1, t1, P.x, curve);
    mont_sub(env, t1, t1, Q.x, curve); // x3 = lambda^2 - x1 - x2
    mont_sub(env, t2, P.x, t1, curve);
    cgbn_set(env, R.x, t1);
    mont_mul(env, t1, lambda, t2, curve);
    mont_sub(env, R.y, t1, P.y, curve); // y3 = lambda*(x1-x3) - y1
    R.z=1;
  }
}

