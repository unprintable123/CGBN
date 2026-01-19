#include <array>
#include <math.h>
#include <unordered_map>
#include <chrono>

#include "ecc/ecc.cu"

constexpr uint32_t DP_SIZE = 15;
constexpr uint32_t DP_MASK = (1 << DP_SIZE) - 1;
constexpr uint32_t MAX_FOUND = 1 << 18;
constexpr uint32_t NB_RUN = 16384;
typedef std::array<uint32_t, 4> uint128_t;

enum Herd {
    Tame = 0,
    Wild = 1,
    InverseWild = 2
};

void from_mpz_signed(cgbn_mem_t<NBITS> &x, mpz_t &src) {
    mpz_t tmp;
    mpz_init_set_ui(tmp, 1);
    if (mpz_sgn(src) < 0) {
        mpz_mul_2exp(tmp, tmp, NBITS);
        mpz_add(tmp, tmp, src);
    } else {
        mpz_set(tmp, src);
    }

    from_mpz(tmp, x._limbs, NBITS/32);
    mpz_clear(tmp);
}

void store_low_128_bits(cgbn_mem_t<NBITS> &x, uint128_t &position) {
    position[0] = x._limbs[0];
    position[1] = x._limbs[1];
    position[2] = x._limbs[2];
    position[3] = x._limbs[3];
}

void load_uint128(uint128_t& dest, mpz_t &src, bool use_sgn=false) {
    bool sgn = dest[3] >> 31;
    mpz_import(src, 4, -1, sizeof(uint32_t), 0, 0, dest.data());
    if (use_sgn && sgn) {
        mpz_t tmp;
        mpz_init_set_ui(tmp, 1);
        mpz_mul_2exp(tmp, tmp, 128);
        mpz_sub(src, tmp, src);
        mpz_neg(src, src);

        mpz_clear(tmp);
    }
}


struct Uint128Hasher {
    std::size_t operator()(const std::array<uint32_t, 4>& arr) const {
        return ((std::size_t)arr[1] << 32) | (std::size_t)arr[2];
    }
};

struct HashEntry
{
    uint128_t offset;
    Herd herd;
};

struct JumpEntry
{
    cgbn_mem_t<NBITS> point;
    cgbn_mem_t<NBITS> offset;
};

struct JumpEntryGPU
{
    env_t::cgbn_t point;
    env_t::cgbn_t offset;
};

ECParameters param;
mpz_t G, G_inv, target, target_inv;
cgbn_mem_t<NBITS> *G_gpu, *G_inv_gpu, *target_gpu, *target_inv_gpu;
JumpEntry *jumptable_gpu;
Curve *gpuCurve;
mpz_t order, max_offset, mid;

struct Kangaroo
{
    cgbn_mem_t<NBITS> P;
    cgbn_mem_t<NBITS> offset;
    Herd herd;
};

__device__ __forceinline__ void mont_pow_gpu(env_t env, env_t::cgbn_t &r, const env_t::cgbn_t &a, const env_t::cgbn_t &n, const CurveGPU &curve) {
    env_t::cgbn_t base, exp;
    cgbn_set(env, base, a);
    cgbn_set(env, exp, n);
    cgbn_set(env, r, curve._r); // montgomery representation of 1

    while (cgbn_equals_ui32(env, exp, 0) == 0) {
        if (cgbn_get_ui32(env, exp) & 1) {
            mont_mul(env, r, r, base, curve);
        }
        mont_sqr(env, base, base, curve);
        cgbn_shift_right(env, exp, exp, 1);
    }
}

struct HashTable
{
    std::unordered_map<uint128_t, HashEntry, Uint128Hasher> entries;
    HashTable() {
        entries.reserve(1000000);
    }

    int addPoint(Kangaroo &point) {
        uint128_t position, offset;
        store_low_128_bits(point.P, position);
        store_low_128_bits(point.offset, offset);
        
        if (entries.find(position) == entries.end()) {
            entries[position] = { offset, point.herd };
            return 0;
        } else {
            auto &entry = entries[position];
            if (point.herd == entry.herd) {
                if (point.herd == Tame) {
                    printf("tame collision detected\n");
                    return -1;
                }
                if (entry.offset == offset) {
                    printf("bad collision detected\n");
                    return -1;
                }
            }
            mpz_t P1, P2;
            mpz_inits(P1, P2, NULL);
            int f1 = compute_node(point.herd, offset, P1);
            int f2 = compute_node(entry.herd, entry.offset, P2);
            if (mpz_cmp(P1, P2) != 0) {
                mpz_clears(P1, P2, NULL);
                return -1;
            }
            mpz_t o1, o2, diff, inv_f;
            mpz_inits(o1, o2, diff, inv_f, NULL);
            load_uint128(offset, o1, true);
            load_uint128(entry.offset, o2, true);
            mpz_sub(diff, o1, o2);

            int f_diff = f2 - f1;
            if (f_diff == 0) {
                printf("bad collision detected2\n");
                mpz_clears(P1, P2, o1, o2, diff, inv_f, NULL);
                return -1;
            }
            mpz_set_si(inv_f, f_diff);
            mpz_invert(inv_f, inv_f, order);
            mpz_mul(diff, diff, inv_f);
            mpz_add(diff, diff, mid);
            mpz_mod(diff, diff, order);

            printf("Solution: 0x");
            print_mpz(diff);
            mpz_clears(P1, P2, o1, o2, diff, inv_f, NULL);
            return 1;
        }
    }

    int compute_node(Herd herd, uint128_t offset, mpz_t &P) {
        mpz_t tmp;
        mpz_init(tmp);
        load_uint128(offset, tmp, true);
        mont_pow(P, G, tmp, param); // P in normal domain
        to_mont(P, P, param); // store as montgomery to match GPU
        int ret;
        switch (herd)
        {
        case Tame:
            ret = 0;
            break;
        case Wild:
            ret = 1;
            mont_mul(P, P, target, param);
            break;
        case InverseWild:
            ret = -1;
            mont_mul(P, P, target_inv, param);
        }
        mpz_clear(tmp);
        return ret;
    }
};

constexpr uint32_t TPI_ONES = (1ull<<TPI)-1;
__device__ __forceinline__ static uint32_t instance_sync_mask() {
    uint32_t group_thread=threadIdx.x & TPI-1, warp_thread=threadIdx.x & warpSize-1;
    return TPI_ONES<<(group_thread ^ warp_thread);
}

__device__ __forceinline__ void save_output(env_t env, Kangaroo *output, uint32_t *output_idx, uint32_t *atom_pos, uint32_t idx, env_t::cgbn_t &P, env_t::cgbn_t &offset, Herd herd) {
    bool is_main = (threadIdx.x % TPI == 0);
    uint32_t pos=4294967295U;
    if (is_main)
        pos = atomicAdd(atom_pos, 1);
    auto sync_mask = instance_sync_mask();
    pos = __shfl_sync(sync_mask, pos, 0, TPI);
    assert(pos < MAX_FOUND);
    cgbn_store(env, &(output[pos].P), P);
    cgbn_store(env, &(output[pos].offset), offset);
    output[pos].herd = herd;
    output_idx[pos] = idx;
}

__global__ void kernel_create_kangaroos(cgbn_error_report_t *report, Kangaroo *kangaroos, cgbn_mem_t<NBITS> *G_, cgbn_mem_t<NBITS> *G_inv_, cgbn_mem_t<NBITS> *target_, cgbn_mem_t<NBITS> *target_inv_, Curve *curve_, size_t count) {
    int32_t instance;
  
    instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(instance>=count)
        return;
    
    context_t          bn_context(cgbn_report_monitor, report, instance);
    env_t              bn_env(bn_context);
    CurveGPU curve;
    cgbn_load(bn_env, curve.p, &(curve_->p));
    cgbn_load(bn_env, curve.a, &(curve_->a));
    cgbn_load(bn_env, curve.b, &(curve_->b));
    cgbn_load(bn_env, curve._r, &(curve_->_r));
    cgbn_load(bn_env, curve._r2, &(curve_->_r2));
    cgbn_load(bn_env, curve._r3, &(curve_->_r3));
    curve.np0 = curve_->np0;

    env_t::cgbn_t G_val, G_inv_val, target_val, target_inv_val;
    cgbn_load(bn_env, G_val, G_);
    cgbn_load(bn_env, G_inv_val, G_inv_);
    cgbn_load(bn_env, target_val, target_);
    cgbn_load(bn_env, target_inv_val, target_inv_);

    env_t::cgbn_t offset, offset_sign, base, P;
    cgbn_load(bn_env, offset, &(kangaroos[instance].offset));
    auto herd = kangaroos[instance].herd;

    cgbn_shift_right(bn_env, offset_sign, offset, NBITS-1);
    if (cgbn_get_ui32(bn_env, offset_sign)>0) {
        cgbn_bitwise_complement(bn_env, offset, offset);
        cgbn_add_ui32(bn_env, offset, offset, 1);
        cgbn_set(bn_env, base, G_inv_val);
    } else {
        cgbn_set(bn_env, base, G_val);
    }

    mont_pow_gpu(bn_env, P, base, offset, curve);

    if (herd == Wild) {
        mont_mul(bn_env, P, P, target_val, curve);
    } else if (herd == InverseWild) {
        mont_mul(bn_env, P, P, target_inv_val, curve);
    }
    cgbn_store(bn_env, &(kangaroos[instance].P), P);
}

__global__ void kernel_jump(cgbn_error_report_t *report, Kangaroo *kangaroos, JumpEntry *jumptable_, Kangaroo *output, uint32_t *output_idx, uint32_t *atom_pos, Curve *curve_, size_t count) {
    int32_t instance;

    instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(instance>=count)
        return;
    
    context_t          bn_context(cgbn_report_monitor, report, instance);
    env_t              bn_env(bn_context);
    CurveGPU curve;
    cgbn_load(bn_env, curve.p, &(curve_->p));
    cgbn_load(bn_env, curve.a, &(curve_->a));
    cgbn_load(bn_env, curve.b, &(curve_->b));
    cgbn_load(bn_env, curve._r, &(curve_->_r));
    cgbn_load(bn_env, curve._r2, &(curve_->_r2));
    cgbn_load(bn_env, curve._r3, &(curve_->_r3));
    curve.np0 = curve_->np0;

    env_t::cgbn_t Ps;
    env_t::cgbn_t offset;
    Herd herd;
    JumpEntryGPU jumptable[32];
    cgbn_load(bn_env, Ps, &(kangaroos[instance].P));
    cgbn_load(bn_env, offset, &(kangaroos[instance].offset));
    herd = kangaroos[instance].herd;

    for (int i = 0; i < 32; i++) {
        cgbn_load(bn_env, jumptable[i].point, &(jumptable_[i].point));
        cgbn_load(bn_env, jumptable[i].offset, &(jumptable_[i].offset));
    }

    for (int nr = 0; nr < NB_RUN; nr++) {
        auto &P = Ps;
        uint32_t jmp = cgbn_get_ui32(bn_env, P) >> (32 - 5);
        mont_mul(bn_env, P, P, jumptable[jmp].point, curve);
        cgbn_add(bn_env, offset, offset, jumptable[jmp].offset);

        if ((cgbn_get_ui32(bn_env, P) & DP_MASK) == 0) {
            save_output(bn_env, output, output_idx, atom_pos, instance, P, offset, herd);
        }
    }

    cgbn_store(bn_env, &(kangaroos[instance].P), Ps);
    cgbn_store(bn_env, &(kangaroos[instance].offset), offset);
}

void input_mpz(const char *name, mpz_t &x) {
    printf("%s: ", name);
    gmp_scanf("%Zi", &x);
}

void read_curve() {
    mpz_inits(G, G_inv, target, target_inv, order, max_offset, mid, NULL);
    mpz_set_ui(param.a, 0);
    mpz_set_ui(param.b, 0);
    input_mpz("p", param.p);
    init_param(param);

    input_mpz("g", G);
    to_mont(G, G, param);
    input_mpz("order", order);
    input_mpz("target", target);
    to_mont(target, target, param);
    input_mpz("bound", max_offset);

    assert(mpz_cmp(max_offset, order) <= 0);
}

void init_jumptable(JumpEntry *jumptable, size_t count)
{
    mpz_t offsets[count];
    mpz_t P;
    mpz_init(P);
    for (size_t i = 0; i < count; i++) {
        mpz_init(offsets[i]);
    }

    double power_range = mpz_get_d(max_offset);
    power_range = log2(power_range) / 2;
    uint32_t jump_bits = ceil(power_range+1);
    printf("jump_bits: %d\n", jump_bits);
    printf("power_range: %f\n", power_range);

    double dist_avg, max_avg, min_avg, max_range;
    max_avg = pow(2.0, power_range + 0.05);
    min_avg = pow(2.0, power_range - 0.05);
    max_range = pow(2.0, power_range + 1);
    while (true) {
        dist_avg = 0.0;
        for (size_t i = 0; i < count; i++) {
            do {
                mpz_randombits(offsets[i], jump_bits);
            } while (mpz_get_d(offsets[i]) > max_range);
            dist_avg += mpz_get_d(offsets[i]);
        }
        dist_avg /= count;
        if (dist_avg <= max_avg && dist_avg >= min_avg) {
            break;
        }
    }
    printf("dist_avg: %f\n", log2(dist_avg));
    for (size_t i = 0; i < count; i++) {
        mont_pow(P, G, offsets[i], param);
        to_mont(P, P, param);
        from_mpz(P, jumptable[i].point._limbs, NBITS/32);
        from_mpz_signed(jumptable[i].offset, offsets[i]);
        mpz_clear(offsets[i]);
    }
    mpz_clear(P);
}

void shuffle_point(Kangaroo &ka, size_t range_size) {
    THREAD_LOCAL_MPZ_WORKSPACE(s, P);
    switch (ka.herd)
    {
    case Tame:
        mpz_randombits(s, range_size, true);
        mont_pow(P, G, s, param);
        to_mont(P, P, param);
        break;
    case Wild:
        mpz_randombits(s, range_size-1, true);
        mont_pow(P, G, s, param);
        to_mont(P, P, param);
        mont_mul(P, P, target, param);
        break;
    case InverseWild:
        mpz_randombits(s, range_size-1, true);
        mont_pow(P, G, s, param);
        to_mont(P, P, param);
        mont_mul(P, P, target_inv, param);
    }
    from_mpz(P, ka.P._limbs, NBITS/32);
    from_mpz_signed(ka.offset, s);
}

Kangaroo *prepare_kangaroo(cgbn_error_report_t *report, size_t range_size, size_t count) {
    mpz_t s;
    mpz_init(s);
    Kangaroo *kangaroos = (Kangaroo *)malloc(count * sizeof(Kangaroo));
    Kangaroo *gpu_kangaroos;
    CUDA_CHECK(cudaMalloc(&gpu_kangaroos, count * sizeof(Kangaroo)));
    for (size_t i = 0; i < count; i++) {
        kangaroos[i].herd = (Herd)(i % 3);
        if (kangaroos[i].herd==Tame) {
            mpz_randombits(s, range_size, true);
        } else {
            mpz_randombits(s, range_size-1, true);
        }
        from_mpz_signed(kangaroos[i].offset, s);
    }
    CUDA_CHECK(cudaMemcpy(gpu_kangaroos, kangaroos, count * sizeof(Kangaroo), cudaMemcpyHostToDevice));
    kernel_create_kangaroos<<<(count+(128/TPI-1))/(128/TPI), 128>>>(report, gpu_kangaroos, G_gpu, G_inv_gpu, target_gpu, target_inv_gpu, gpuCurve, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    free(kangaroos);
    mpz_clear(s);
    return gpu_kangaroos;
}

void init() {
    mpz_t shift, shift_inv;
    mpz_inits(shift, shift_inv, NULL);
    mpz_div_2exp(mid, max_offset, 1); // mid = max_offset / 2

    mont_pow(shift, G, mid, param);
    to_mont(shift, shift, param);
    mont_inv(shift_inv, shift, param);
    mont_mul(target, target, shift_inv, param); // target = target * g^{-mid}
    mont_inv(target_inv, target, param);
    mont_inv(G_inv, G, param);

    cgbn_mem_t<NBITS> G_cgbn, G_inv_cgbn, target_cgbn, target_inv_cgbn;
    from_mpz(G, G_cgbn._limbs, NBITS/32);
    from_mpz(G_inv, G_inv_cgbn._limbs, NBITS/32);
    from_mpz(target, target_cgbn._limbs, NBITS/32);
    from_mpz(target_inv, target_inv_cgbn._limbs, NBITS/32);

    CUDA_CHECK(cudaMalloc(&G_gpu, sizeof(cgbn_mem_t<NBITS>)));
    CUDA_CHECK(cudaMemcpy(G_gpu, &G_cgbn, sizeof(cgbn_mem_t<NBITS>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&G_inv_gpu, sizeof(cgbn_mem_t<NBITS>)));
    CUDA_CHECK(cudaMemcpy(G_inv_gpu, &G_inv_cgbn, sizeof(cgbn_mem_t<NBITS>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&target_gpu, sizeof(cgbn_mem_t<NBITS>)));
    CUDA_CHECK(cudaMemcpy(target_gpu, &target_cgbn, sizeof(cgbn_mem_t<NBITS>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&target_inv_gpu, sizeof(cgbn_mem_t<NBITS>)));
    CUDA_CHECK(cudaMemcpy(target_inv_gpu, &target_inv_cgbn, sizeof(cgbn_mem_t<NBITS>), cudaMemcpyHostToDevice));

    mpz_clears(shift, shift_inv, NULL);
}


int main()
{
    HashTable table;
    JumpEntry jumptable[64];
    cgbn_error_report_t *report;
    read_curve();

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    auto sm_count = prop.multiProcessorCount * 2;
    uint32_t TPB = 128;
    auto num_kangaroo = sm_count * (TPB / TPI);
    printf("num_kangaroo: %d, tpb: %d\n", num_kangaroo, TPB);
    init();
    init_jumptable(jumptable, 32);

    CUDA_CHECK(cudaMalloc(&jumptable_gpu, 64 * sizeof(JumpEntry)));
    CUDA_CHECK(cudaMemcpy(jumptable_gpu, jumptable, 64 * sizeof(JumpEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    Curve *curve = new Curve(param);
    CUDA_CHECK(cudaMalloc(&gpuCurve, sizeof(Curve)));
    CUDA_CHECK(cudaMemcpy(gpuCurve, curve, sizeof(Curve), cudaMemcpyHostToDevice));

    size_t range_size = mpz_sizeinbase(mid, 2);
    Kangaroo *kangaroos = prepare_kangaroo(report, range_size, num_kangaroo);

    // {
    //     Kangaroo *cpu_verify = (Kangaroo *)malloc(num_kangaroo * sizeof(Kangaroo));
    //     CUDA_CHECK(cudaMemcpy(cpu_verify, kangaroos, num_kangaroo * sizeof(Kangaroo), cudaMemcpyDeviceToHost));
    //     mpz_t P_expected, P_got;
    //     mpz_inits(P_expected, P_got, NULL);
    //     uint128_t off;
    //     for (size_t i = 0; i < num_kangaroo; i++) {
    //         store_low_128_bits(cpu_verify[i].offset, off);
    //         table.compute_node(cpu_verify[i].herd, off, P_expected);
    //         to_mpz(P_got, cpu_verify[i].P._limbs, NBITS/32);
    //         if (mpz_cmp(P_expected, P_got) != 0) {
    //             printf("cpu verify failed idx %lu herd %d\n", i, cpu_verify[i].herd);
    //             print_mpz(P_expected);
    //             print_mpz(P_got);
    //             exit(1);
    //         }
    //     }
    //     mpz_clears(P_expected, P_got, NULL);
    //     free(cpu_verify);
    //     printf("cpu verify ok\n");
    // }

    Kangaroo *gpuOutput, *output;
    uint32_t atom_pos;
    uint32_t *gpu_atom_pos;
    uint32_t *output_idx, *gpu_output_idx;
    output = (Kangaroo *)malloc(MAX_FOUND * sizeof(Kangaroo));
    output_idx = (uint32_t *)malloc(MAX_FOUND * sizeof(uint32_t));
    CUDA_CHECK(cudaMalloc(&gpuOutput, MAX_FOUND * sizeof(Kangaroo)));
    CUDA_CHECK(cudaMalloc(&gpu_output_idx, MAX_FOUND * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&gpu_atom_pos, sizeof(uint32_t)));

    auto start_time = std::chrono::high_resolution_clock::now();
    bool find=false;
    size_t cnt=0;
    while(!find) {
        CUDA_CHECK(cudaMemset(gpu_atom_pos, 0, sizeof(uint32_t)));
        kernel_jump<<<sm_count, TPB>>>(report, kangaroos, jumptable_gpu, gpuOutput, gpu_output_idx, gpu_atom_pos, gpuCurve, num_kangaroo);
        CUDA_CHECK(cudaDeviceSynchronize());
        CGBN_CHECK(report);
        CUDA_CHECK(cudaMemcpy(&atom_pos, gpu_atom_pos, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output, gpuOutput, atom_pos * sizeof(Kangaroo), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output_idx, gpu_output_idx, atom_pos * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < atom_pos; i++) {
            auto ret = table.addPoint(output[i]);
            if (ret == 1) {
                find = true;
                break;
            } else if (ret != 0) {
                Kangaroo ka;
                ka.herd = output[i].herd;
                shuffle_point(ka, range_size);
                CUDA_CHECK(cudaMemcpy(kangaroos+output_idx[i], &ka, sizeof(Kangaroo), cudaMemcpyHostToDevice));
            }
        }
        cnt += num_kangaroo * NB_RUN;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    printf("cnt: %lu\n", cnt);
    printf("table size: %lu\n", table.entries.size());
    {
        std::chrono::duration<double> elapsed = end_time - start_time;
        double miliseconds = elapsed.count() * 1000;
        printf("time: %.2f ms\n", miliseconds);
        double ops_per_ms = ((double)cnt / miliseconds) / 1000.0;
        printf("Mops/s: %.2f\n", ops_per_ms);
    }
}
