/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/


/**************************************************************************
 * Addition
 **************************************************************************/
template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_add(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t LOOPS=LOOP_COUNT(bits, xt_add);
  bn_t    x0, x1, r;
    
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));

  _env.set(r, x0);
  #pragma unroll 10
  for(int32_t loop=0;loop<LOOPS;loop++)
    _env.add(r, r, x1);

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_add_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_add(instances);
}


/**************************************************************************
 * Subtraction
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_sub(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t LOOPS=LOOP_COUNT(bits, xt_sub);
  bn_t    x0, x1, r;
    
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));

  _env.set(r, x0);
  #pragma unroll 10
  for(int32_t loop=0;loop<LOOPS;loop++)
    _env.sub(r, r, x1);

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_sub_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_sub(instances);
}


/**************************************************************************
 * Accumulate
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_accumulate(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t          LOOPS=LOOP_COUNT(bits, xt_accumulate);
  bn_t             x0, x1, r;
  bn_accumulator_t acc;
  
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));

  _env.set(acc, x0);
  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++)
    _env.add(acc, x1);
  _env.resolve(r, acc);

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_accumulate_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_accumulate(instances);
}


/**************************************************************************
 * Multiplication
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_mul(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_mul);
  bn_t      x0, x1, r;
  bn_wide_t w;

  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));

  _env.set(r, x0);
  #pragma unroll 4
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.mul_wide(w, r, x0);
    _env.set(r, w._low);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_mul_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_mul(instances);
}


/**************************************************************************
 * Division
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_div_qr(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_div_qr);
  bn_t      x0, x1, q, r;
  bn_wide_t w;
  
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));

  _env.shift_right(w._high, x1, 1);
  _env.set(w._low, x0);
  
  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.div_rem_wide(q, r, w, x1);
    _env.set(w._high, r);
    _env.set(w._low, q);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_div_qr_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_div_qr(instances);
}

/**************************************************************************
 * Square root
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_sqrt(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_sqrt);
  bn_t      r;
  bn_wide_t w;
  
  _env.load(w._low, &(instances[_instance].x0));
  _env.load(w._high, &(instances[_instance].x1));

  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.sqrt_wide(r, w);
    _env.set(w._low, r);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_sqrt_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_sqrt(instances);
}


/**************************************************************************
 * Mont reduce
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_mont_reduce(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_mont_reduce);
  bn_t      x0, x1, o0, r;
  bn_wide_t w;
  uint32_t  mp0;
  
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));
  _env.load(o0, &(instances[_instance].o0));
  
  _env.set(w._low, x0);
  _env.set(w._high, x1);

  mp0=-_env.binary_inverse_ui32(_env.get_ui32(o0));
  
  #pragma nounrull
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.mont_reduce_wide(r, w, o0, mp0);
    _env.set(w._low, r);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_mont_reduce_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_mont_reduce(instances);
}


/**************************************************************************
 * GCD
 **************************************************************************/

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_gcd(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_gcd);
  bn_t      x0, x1, r;
  
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));
  
  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++) {
    _env.gcd(r, x0, x1);
    _env.add_ui32(x0, x0, 1);
    _env.add_ui32(x1, x1, 1);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_gcd_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_gcd(instances);
}


/**************************************************************************
 * Mod Inv
 **************************************************************************/

#define DIVSTEPS_N 30
#include "cgbn/arith/arith.h"

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
    uint32_t w = (g * NEGINV16[(f&15)]) & ((1<<limit) - 1); // TODO: use ubfe
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


template<class env_t>
__device__ __forceinline__ void x_modinv_odd_faster(env_t bn_env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &x, const typename env_t::cgbn_t &m) {
  typename env_t::cgbn_t      s, u, v;                                       // define m, r, s, u, v as 1024-bit bignums
  typename env_t::cgbn_wide_t w;                                                   // define w as a wide (2048-bit) bignum
  uint32_t           np0;
  int32_t            k=0, carry, compare;

  cgbn_set(bn_env, u, m);
  cgbn_set(bn_env, v, x);
  cgbn_set_ui32(bn_env, r, 0);
  cgbn_set_ui32(bn_env, s, 1);
  
  while(true) {
    k++;
    if(cgbn_get_ui32(bn_env, u)%2==0) {
      cgbn_rotate_right(bn_env, u, u, 1);
      cgbn_add(bn_env, s, s, s);
    }
    else if(cgbn_get_ui32(bn_env, v)%2==0) {
      cgbn_rotate_right(bn_env, v, v, 1);
      cgbn_add(bn_env, r, r, r);
    }
    else {
      compare=cgbn_compare(bn_env, u, v);
      if(compare>0) {
        cgbn_add(bn_env, r, r, s);
        cgbn_sub(bn_env, u, u, v);
        cgbn_rotate_right(bn_env, u, u, 1);
        cgbn_add(bn_env, s, s, s);
      }
      else if(compare<0) {
        cgbn_add(bn_env, s, s, r);
        cgbn_sub(bn_env, v, v, u);
        cgbn_rotate_right(bn_env, v, v, 1);
        cgbn_add(bn_env, r, r, r);
      }
      else
        break;
    }
  }
  
  if(!cgbn_equals_ui32(bn_env, u, 1)) 
    cgbn_set_ui32(bn_env, r, 0);
  else {  
    // last r update
    carry=cgbn_add(bn_env, r, r, r);
    if(carry==1) 
      cgbn_sub(bn_env, r, r, m);

    // clean up
    if(cgbn_compare(bn_env, r, m)>0)
      cgbn_sub(bn_env, r, r, m);
    cgbn_sub(bn_env, r, m, r);

    // faster cleanup, taking advantage of the built-in mont_reduce_wide.
    np0=-cgbn_binary_inverse_ui32(bn_env, cgbn_get_ui32(bn_env, m));
    cgbn_set(bn_env, w._low, r);
    cgbn_set_ui32(bn_env, w._high, 0);    
    cgbn_mont_reduce_wide(bn_env, r, w, m, np0);

    cgbn_shift_left(bn_env, w._low, r, 2*env_t::BITS-k);
    cgbn_shift_right(bn_env, w._high, r, k-env_t::BITS);
    cgbn_mont_reduce_wide(bn_env, r, w, m, np0);
  }
}

template<uint32_t tpi, uint32_t bits>
__device__ __forceinline__ void xmp_tester<tpi, bits>::x_test_modinv(xmp_tester<tpi, bits>::x_instance_t *instances) {
  int32_t   LOOPS=LOOP_COUNT(bits, xt_modinv);
  bn_t      x0, x1, r;
  
  _env.load(x0, &(instances[_instance].x0));
  _env.load(x1, &(instances[_instance].x1));
  
  #pragma nounroll
  for(int32_t loop=0;loop<LOOPS;loop++) {
    // _env.modular_inverse(r, x0, x1);
    cgbn_bitwise_mask_ior(_env, x1, x1, 1);
    // x_modinv_odd_faster(_env, r, x0, x1);
    yang_modinv_odd(_env, r, x0, x1);
    _env.add_ui32(x0, x0, 1);
    _env.add_ui32(x1, x1, 1);
  }

  _env.store(&(instances[_instance].r), r);
}

template<uint32_t tpi, uint32_t bits>
__global__ void x_test_modinv_kernel(typename xmp_tester<tpi, bits>::x_instance_t *instances, uint32_t count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/tpi;
  
  if(instance>=count)
    return;
  
  xmp_tester<tpi, bits> tester(cgbn_no_checks, NULL, instance);
  tester.x_test_modinv(instances);
}
