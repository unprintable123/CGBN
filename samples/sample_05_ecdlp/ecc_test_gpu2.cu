#include "ecc/ecc.cu"

#define INSTANCES 128000

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
  point_mul(Q, G, 0x760532837443, param);

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
    point_add(P, P, Q, param);
    if (!(P==R)) {
      printf("gpu kernel failed on instance %d\n", i);
      return;
    }
  }
}

__global__ void kernel_point_add(cgbn_error_report_t *report, instance_t *instances, Curve *curve_, uint32_t count) {
  int32_t instance;
  
  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count)
    return;

  context_t          bn_context(cgbn_report_monitor, report, instance);   // construct a context
  env_t              bn_env(bn_context);                                  // construct an environment for 1024-bit math
  ECPointGPU A[GRP_INV_SIZE], B[GRP_INV_SIZE];
  CurveGPU curve;
  cgbn_load(bn_env, curve.p, &(curve_->p));
  cgbn_load(bn_env, curve.a, &(curve_->a));
  cgbn_load(bn_env, curve.b, &(curve_->b));
  cgbn_load(bn_env, curve._r, &(curve_->_r));
  cgbn_load(bn_env, curve._r2, &(curve_->_r2));
  cgbn_load(bn_env, curve._r3, &(curve_->_r3));
  curve.np0 = curve_->np0;

  for (int i = 0; i < GRP_INV_SIZE; i++) {
    load_point(bn_env, A[i], &(instances[instance*GRP_INV_SIZE+i].A));
    load_point(bn_env, B[i], &(instances[instance*GRP_INV_SIZE+i].B));
  }

  batch_point_add(bn_env, A, A, B, curve);

  for (int i = 0; i < GRP_INV_SIZE; i++)
    save_point(bn_env, &(instances[instance*GRP_INV_SIZE+i].C), A[i]);
}


int main() {
  static_assert(INSTANCES % GRP_INV_SIZE == 0, "GRP_INV_SIZE must divide INSTANCES");
  read_curve();
  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  Curve *curve = new Curve(param);
  Curve *gpuCurve;
  
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
  kernel_point_add<<<((INSTANCES/GRP_INV_SIZE)+(128/TPI-1))/(128/TPI), 128>>>(report, gpuInstances, gpuCurve, INSTANCES/GRP_INV_SIZE);

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



