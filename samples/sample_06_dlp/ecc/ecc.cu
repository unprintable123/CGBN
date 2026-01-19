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

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, NBITS> env_t;

typedef struct {
    env_t::cgbn_t p;
    env_t::cgbn_t a;
    env_t::cgbn_t b;
    env_t::cgbn_t _r;
    env_t::cgbn_t _r2;
    env_t::cgbn_t _r3;
    uint32_t np0;
} CurveGPU;

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
