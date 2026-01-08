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

#define INSTANCES 100000
#define GRP_INV_SIZE 128
#define ADD_CHECK true

struct Curve {
    cgbn_mem_t<NBITS> p;
    cgbn_mem_t<NBITS> a;
    cgbn_mem_t<NBITS> b;
    cgbn_mem_t<NBITS> _r2;
    cgbn_mem_t<NBITS> _r3;
    uint32_t np0;
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

ECParameters param;
ECPoint G, target, target_inv;
mpz_t order, max_offset, mid;

void read_curve() {
    mpz_inits(order, max_offset, mid, NULL);
    // y^2 = x^3 + 7
    mpz_set_ui(param.a, 0);
    mpz_set_ui(param.b, 7);
    mpz_set_str(param.p, "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f", 16);
    init_param(param);

    mpz_set_str(G.x, "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", 16);
    mpz_set_str(G.y, "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8", 16);
    mpz_set_str(order, "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141", 16);
    G.to_mont_(param);

    mpz_set_str(max_offset, "fffffffffff", 16);
    point_mul(target, G, 0xa87c356f637L, param);

    assert(mpz_cmp(max_offset, order) < 0);
}

instance_t *generate_instances(uint32_t count) {
  instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);

  ECPoint P, Q;
  point_mul(P, G, 0x123, param);
  point_mul(Q, G, 0x96532837443, param);

  for (uint32_t i=0; i<count; i++) {
    instances[i].A.from_point(P);
    point_add(P, P, G, param);
    instances[i].B.from_point(Q);
    point_add(Q, Q, P, param);
  }
  return instances;
}

void verify_results(instance_t *instances, uint32_t count) {
  ECPoint P, Q, R;
  for (uint32_t i=0; i<count; i++) {
    instances[i].A.to_point(P);
    instances[i].B.to_point(Q);
    instances[i].C.to_point(R);
    point_double(P, P, param);
    point_add(P, P, Q, param);
    if (!(P==R)) {
      printf("gpu kernel failed on instance %d\n", i);
      return;
    }
  }
}

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
    env_t::cgbn_t _r2;
    env_t::cgbn_t _r3;
    uint32_t np0;
} CurveGPU;

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
  if (carry)
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
}

template<class env_t>
__device__ __forceinline__ void mont_sqr(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const CurveGPU &curve) {
  cgbn_mont_sqr(env, r, a, curve.p, curve.np0);
}

template<class env_t>
__device__ __forceinline__ void mont_inv(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a, const CurveGPU &curve) {
  // cgbn_modular_inverse(env, r, a, curve.p);
  yang_modinv_odd(env, r, a, curve.p);
  cgbn_mont_mul(env, r, r, curve._r3, curve.p, curve.np0);
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


__global__ void kernel_point_add(cgbn_error_report_t *report, instance_t *instances, Curve *curve_, uint32_t count) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t          bn_context(cgbn_report_monitor, report, instance);   // construct a context
  env_t              bn_env(bn_context);                                  // construct an environment for 1024-bit math
  ECPointGPU A, B;
  CurveGPU curve;
  cgbn_load(bn_env, curve.p, &(curve_->p));
  cgbn_load(bn_env, curve.a, &(curve_->a));
  cgbn_load(bn_env, curve.b, &(curve_->b));
  cgbn_load(bn_env, curve._r2, &(curve_->_r2));
  cgbn_load(bn_env, curve._r3, &(curve_->_r3));
  curve.np0 = curve_->np0;

  load_point(bn_env, A, &(instances[instance].A));
  load_point(bn_env, B, &(instances[instance].B));

  point_double(bn_env, A, A, curve);
  point_add(bn_env, A, A, B, curve);

  save_point(bn_env, &(instances[instance].C), A);
}





int main() {
  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  Curve *curve = new Curve;
  Curve *gpuCurve;

  read_curve();
  from_mpz(param.p, curve->p._limbs, NBITS/32);
  from_mpz(param._a_mont, curve->a._limbs, NBITS/32);
  from_mpz(param._b_mont, curve->b._limbs, NBITS/32);
  from_mpz(param._r2, curve->_r2._limbs, NBITS/32);
  from_mpz(param._r3, curve->_r3._limbs, NBITS/32);
  curve->np0 = mpz_get_ui(param._p_inv_neg);
  
  printf("Genereating instances ...\n");
  instances=generate_instances(INSTANCES);
  
  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*INSTANCES));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*INSTANCES, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc((void **)&gpuCurve, sizeof(Curve)));
  CUDA_CHECK(cudaMemcpy(gpuCurve, curve, sizeof(Curve), cudaMemcpyHostToDevice));
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  auto start = std::chrono::steady_clock::now();
  
  printf("Running GPU kernel ...\n");
  // launch with 32 threads per instance, 128 threads (4 instances) per block
  kernel_point_add<<<(INSTANCES+3)/4, 128>>>(report, gpuInstances, gpuCurve, INSTANCES);

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



