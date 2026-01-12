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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <chrono>
#include "cgbn/cgbn.h"
#include "cgbn/arith/arith.h"
#include "../utility/support.h"

/************************************************************************************************
 *  This modinv example is based on the Right-Shift Algorithm for Classical Modular Inverse
 *  from the paper "New Algorithm for Classical Modular Inverse" by Robert Lorencz.
 *
 *  This algorithm only works for an odd modulus.
 *
 *  The clean-up phase uses the XMP library built-in for Montgomery reductions, which improves
 *  performance quite a bit.  
 ************************************************************************************************/
 

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 4
#define BITS 256
#define INSTANCES 100000

// Declare the instance type
typedef struct {
  cgbn_mem_t<BITS> x;
  cgbn_mem_t<BITS> m;
  cgbn_mem_t<BITS> inverse;
} instance_t;

// support routine to generate random instances
instance_t *generate_instances(uint32_t count) {
  auto seed=time(NULL);
  printf("seed=%ld\n", seed);
  srand(seed);
  instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);

  for(int index=0;index<count;index++) {
    random_words(instances[index].x._limbs, BITS/32);
    random_words(instances[index].m._limbs, BITS/32);
    instances[index].m._limbs[0] |= 1;                 // guarantee modulus is odd
  }
  return instances;
}

// support routine to verify the GPU results using the CPU
void verify_results(instance_t *instances, uint32_t count) {
  mpz_t x, m, computed, correct;
  
  mpz_init(x);
  mpz_init(m);
  mpz_init(computed);
  mpz_init(correct);
  
  for(int index=0;index<count;index++) {
    to_mpz(x, instances[index].x._limbs, BITS/32);
    to_mpz(m, instances[index].m._limbs, BITS/32);
    to_mpz(computed, instances[index].inverse._limbs, BITS/32);
    
    if(mpz_invert(correct, x, m)==0)
      mpz_set_ui(correct, 0);
    else {
      if(mpz_cmp(correct, computed)!=0) {
        printf("gpu inverse kernel failed on instance %d\n", index);
        return;
      }
    }
  }
  
  mpz_clear(x);
  mpz_clear(m);
  mpz_clear(computed);
  mpz_clear(correct);
  
  printf("All results match\n");
}

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

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

    cgbn_shift_left(bn_env, w._low, r, 2*BITS-k);
    cgbn_shift_right(bn_env, w._high, r, k-BITS);
    cgbn_mont_reduce_wide(bn_env, r, w, m, np0);
  }
}

// the actual kernel
__global__ void kernel_modinv_odd(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t          bn_context(cgbn_report_monitor, report, instance);   // construct a context
  env_t              bn_env(bn_context);                                  // construct an environment for 1024-bit math
  env_t::cgbn_t      m, r, v;                                       // define m, r, s, u, v as 1024-bit bignums
  // env_t::cgbn_wide_t w;                                                   // define w as a wide (2048-bit) bignum
  // uint32_t           np0;
  // int32_t            k=0, carry, compare;

  cgbn_load(bn_env, m, &(instances[instance].m));
  cgbn_load(bn_env, v, &(instances[instance].x));

  for (int i = 0; i < 5000; i++){
    yang_modinv_odd(bn_env, r, v, m);
    // x_modinv_odd_faster(bn_env, r, v, m);
    // cgbn_modular_inverse(bn_env, r, v, m);
  }

//   cgbn_set(bn_env, u, m);
//   cgbn_set_ui32(bn_env, r, 0);
//   cgbn_set_ui32(bn_env, s, 1);
  
//   while(true) {
//     k++;
//     if(cgbn_get_ui32(bn_env, u)%2==0) {
//       cgbn_rotate_right(bn_env, u, u, 1);
//       cgbn_add(bn_env, s, s, s);
//     }
//     else if(cgbn_get_ui32(bn_env, v)%2==0) {
//       cgbn_rotate_right(bn_env, v, v, 1);
//       cgbn_add(bn_env, r, r, r);
//     }
//     else {
//       compare=cgbn_compare(bn_env, u, v);
//       if(compare>0) {
//         cgbn_add(bn_env, r, r, s);
//         cgbn_sub(bn_env, u, u, v);
//         cgbn_rotate_right(bn_env, u, u, 1);
//         cgbn_add(bn_env, s, s, s);
//       }
//       else if(compare<0) {
//         cgbn_add(bn_env, s, s, r);
//         cgbn_sub(bn_env, v, v, u);
//         cgbn_rotate_right(bn_env, v, v, 1);
//         cgbn_add(bn_env, r, r, r);
//       }
//       else
//         break;
//     }
//   }
  
//   if(!cgbn_equals_ui32(bn_env, u, 1)) 
//     cgbn_set_ui32(bn_env, r, 0);
//   else {  
//     // last r update
//     carry=cgbn_add(bn_env, r, r, r);
//     if(carry==1) 
//       cgbn_sub(bn_env, r, r, m);

//     // clean up
//     if(cgbn_compare(bn_env, r, m)>0)
//       cgbn_sub(bn_env, r, r, m);
//     cgbn_sub(bn_env, r, m, r);

//     // faster cleanup, taking advantage of the built-in mont_reduce_wide.
//     np0=-cgbn_binary_inverse_ui32(bn_env, cgbn_get_ui32(bn_env, m));
//     cgbn_set(bn_env, w._low, r);
//     cgbn_set_ui32(bn_env, w._high, 0);    
//     cgbn_mont_reduce_wide(bn_env, r, w, m, np0);

//     cgbn_shift_left(bn_env, w._low, r, 2*BITS-k);
//     cgbn_shift_right(bn_env, w._high, r, k-BITS);
//     cgbn_mont_reduce_wide(bn_env, r, w, m, np0);
//   }

  cgbn_store(bn_env, &(instances[instance].inverse), r);
}

int main() {
  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  
  printf("Genereating instances ...\n");
  instances=generate_instances(INSTANCES);
  
  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*INSTANCES));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*INSTANCES, cudaMemcpyHostToDevice));
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  auto start = std::chrono::steady_clock::now();
  
  printf("Running GPU kernel ...\n");
  // launch with 32 threads per instance, 128 threads (4 instances) per block
  kernel_modinv_odd<<<(INSTANCES+3)/4, 128>>>(report, gpuInstances, INSTANCES);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  auto end = std::chrono::steady_clock::now();
  printf("GPU kernel completed in %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0);

    
  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*INSTANCES, cudaMemcpyDeviceToHost));
  
  printf("Verifying the results ...\n");
  verify_results(instances, INSTANCES);
  
  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}
