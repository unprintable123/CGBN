#include "cuda.h"
#include "cgbn/cgbn.h"
#include "cgbn/arith/arith.h"

#define DIVSTEPS_N 30

__device__ __forceinline__ int32_t divsteps_n_matrix_var(int32_t eta, uint32_t f, uint32_t g, int32_t &u, int32_t &v, int32_t &q, int32_t &r) {
  static const unsigned char NEGINV16[16] = {0, 15, 0, 5, 0, 3, 0, 9, 0, 7, 0, 13, 0, 11, 0, 1};
  uint32_t i = DIVSTEPS_N;
  u = 1;
  v = 0;
  q = 0;
  r = 1;
  while (true)
  {
    uint32_t zeros = umin(i, cgbn::uctz(g));
    eta -= zeros;
    i -= zeros;
    g >>= zeros;
    u = (int32_t)((uint32_t)u << zeros);
    v = (int32_t)((uint32_t)v << zeros);
    if (i == 0)
      break;
    if (eta < 0) {
      // eta, f, u, v, g, q, r = -eta, g, q, r, -f, -u, -v
      eta = -eta;
      int32_t tmp = f;
      f = g;
      g = (uint32_t)(-tmp);

      tmp = u;
      u = q;
      q = -tmp;
      
      tmp = v;
      v = r;
      r = -tmp;
    }
    uint32_t limit = min(min(eta+1, (int32_t)i), 4);
    uint32_t w = (g * NEGINV16[(f&15)]) & ((1<<limit) - 1);
    // g, q, r = g + w*f, q + w*u, r + w*v
    g = g + w*f;
    q = q + int32_t(w)*u;
    r = r + int32_t(w)*v;
  }
  return eta;
}

template<class env_t>
__device__ __forceinline__ int32_t signed_mul_i32(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const int32_t a_high, const int32_t b) {
  int32_t b_abs = (b < 0) ? -b : b;

  int32_t r_high = a_high * b_abs + (int32_t)cgbn_mul_ui32(env, r, a, b_abs);
  if (b < 0) {
    r_high = ~r_high;
    cgbn_bitwise_complement(env, r, r);
    r_high += cgbn_add_ui32(env, r, r, 1U);
  }
  return r_high;
}

template<class env_t>
__device__ __forceinline__ void yang_modinv_odd(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const typename env_t::cgbn_t &p) {
  // returns r = a^-1 mod p, p must be odd
  typename env_t::cgbn_t f, g, e, c1, c2, tmp;
  auto &d = r;

  // TODO: move it outside of the function
  uint32_t pi = cgbn_binary_inverse_ui32(env, cgbn_get_ui32(env, p)) & ((1<<DIVSTEPS_N)-1); // pi * p = 1 mod 2^DIVSTEPS_N

  int32_t eta = -1, f_high = 0, g_high = 0, d_high = 0, e_high = 0;

  env.set(f, p);
  env.set(g, a);
  env.set_ui32(d, 0);
  env.set_ui32(e, 1);

  while(env.equals_ui32(g, 0)==0) {
    int32_t u, v, q, r;
    eta = divsteps_n_matrix_var(eta, cgbn_get_ui32(env, f), cgbn_get_ui32(env, g), u, v, q, r);
    { // update f,g
      int32_t cf_high1 = signed_mul_i32(env, c1, f, f_high, u);
      int32_t cf_high2 = signed_mul_i32(env, tmp, g, g_high, v);
      int32_t cf_high = cf_high1 + cf_high2 + cgbn_add(env, c1, c1, tmp);

      int32_t cg_high1 = signed_mul_i32(env, c2, f, f_high, q);
      int32_t cg_high2 = signed_mul_i32(env, tmp, g, g_high, r);
      int32_t cg_high = cg_high1 + cg_high2 + cgbn_add(env, c2, c2, tmp);

      f_high = cgbn_shift_right_extend_signed<DIVSTEPS_N>(env, f, c1, cf_high);
      g_high = cgbn_shift_right_extend_signed<DIVSTEPS_N>(env, g, c2, cg_high);
    }
    { // update d,e
      int32_t md, me;
      uint32_t d_low = cgbn_get_ui32(env, d);
      uint32_t e_low = cgbn_get_ui32(env, e);
      
      md = (d_high < 0 ? u : 0) + (e_high < 0 ? v : 0);
      me = (d_high < 0 ? q : 0) + (e_high < 0 ? r : 0);

      uint32_t cd_low = uint32_t(u) * d_low + uint32_t(v) * e_low;
      uint32_t ce_low = uint32_t(q) * d_low + uint32_t(r) * e_low;
      md -= int32_t((cd_low * pi + uint32_t(md)) & ((1<<DIVSTEPS_N)-1));
      me -= int32_t((ce_low * pi + uint32_t(me)) & ((1<<DIVSTEPS_N)-1));

      // cd, ce = u*d + v*e + p*md, q*d + r*e + p*me
      int32_t cd_high1 = signed_mul_i32(env, c1, d, d_high, u);
      int32_t cd_high2 = signed_mul_i32(env, tmp, e, e_high, v);
      int32_t cd_high = cd_high1 + cd_high2 + cgbn_add(env, c1, c1, tmp);
      int32_t cd_high3 = signed_mul_i32(env, tmp, p, (int32_t)0, md);
      cd_high += cd_high3 + cgbn_add(env, c1, c1, tmp);

      int32_t ce_high1 = signed_mul_i32(env, c2, d, d_high, q);
      int32_t ce_high2 = signed_mul_i32(env, tmp, e, e_high, r);
      int32_t ce_high = ce_high1 + ce_high2 + cgbn_add(env, c2, c2, tmp);
      int32_t ce_high3 = signed_mul_i32(env, tmp, p, (int32_t)0, me);
      ce_high += ce_high3 + cgbn_add(env, c2, c2, tmp);

      d_high = cgbn_shift_right_extend_signed<DIVSTEPS_N>(env, d, c1, cd_high);
      e_high = cgbn_shift_right_extend_signed<DIVSTEPS_N>(env, e, c2, ce_high);
    }
  }
  if (d_high < 0) {
    d_high += (int32_t)cgbn_add(env, d, d, p);
  }
  if (f_high < 0) {
    d_high = ~d_high;
    cgbn_bitwise_complement(env, d, d);
    d_high += cgbn_add_ui32(env, d, d, 1U);
  }
  if (d_high < 0) {
    d_high += (int32_t)cgbn_add(env, d, d, p);
  }
  // env.set(r, d);
}
