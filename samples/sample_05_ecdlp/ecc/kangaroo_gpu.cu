#include <array>
#include <math.h>
#include <unordered_map>

#include "ecc.cu"

constexpr uint32_t DP_SIZE = 15;
constexpr uint32_t DP_MASK = (1 << DP_SIZE) - 1;
constexpr uint32_t MAX_FOUND = 1 << 18;
constexpr uint32_t NB_RUN = 512;
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

void store_low_256_bits(cgbn_mem_t<NBITS> &x, uint128_t &position) {
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
        std::size_t h = 0;
        for (uint32_t x : arr) {
            h ^= std::hash<uint32_t>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct HashEntry
{
    uint128_t offset;
    Herd herd;
};

struct JumpEntry
{
    ECPointCGBN point;
    cgbn_mem_t<NBITS> offset;
};

struct JumpEntryGPU
{
    ECPointGPU point;
    env_t::cgbn_t offset;
};

ECParameters param;
ECPoint G, target, target_inv;
ECPointCGBN *G_gpu, *target_gpu, *target_inv_gpu;
JumpEntry *jumptable_gpu;
Curve *gpuCurve;
mpz_t order, max_offset, mid;

struct Kangaroo
{
    ECPointCGBN P;
    cgbn_mem_t<NBITS> offset;
    Herd herd;
};

struct HashTable
{
    std::unordered_map<uint128_t, HashEntry, Uint128Hasher> entries;
    HashTable() {
        entries.reserve(1000000);
    }

    int addPoint(Kangaroo &point) {
        uint128_t position, offset;
        store_low_256_bits(point.P.x, position);
        store_low_256_bits(point.offset, offset);
        
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
            printf("%d %d\n", point.herd, entry.herd);
            ECPoint P1, P2;
            int f1 = compute_node(point.herd, offset, P1);
            int f2 = compute_node(entry.herd, entry.offset, P2);
            assert(mpz_cmp(P1.x, P2.x) == 0);
            mpz_t o1, o2;
            mpz_inits(o1, o2, NULL);
            load_uint128(offset, o1, true);
            load_uint128(entry.offset, o2, true);
            if(mpz_cmp(P1.y, P2.y)==0) {
                f1 -= f2;
                mpz_sub(o1, o1, o2);
            } else {
                f1 += f2;
                mpz_add(o1, o1, o2);
            }
            if (f1 == 0) {
                printf("bad collision detected2\n");
                mpz_clears(o1, o2, NULL);
                return -1;
            }
            mpz_set_si(o2, -f1);
            mpz_invert(o2, o2, order);
            mpz_mul(o1, o1, o2);
            mpz_add(o1, o1, mid);
            mpz_mod(o1, o1, order);
            
            printf("Solution: 0x");
            print_mpz(o1);
            mpz_clears(o1, o2, NULL);
            return 1;
        }
    }

    int compute_node(Herd herd, uint128_t offset, ECPoint &P) {
        mpz_t tmp;
        mpz_init(tmp);
        load_uint128(offset, tmp, true);
        point_mul(P, G, tmp, param);
        mpz_clear(tmp);
        int ret;
        switch (herd)
        {
        case Tame:
            ret = 0;
            break;
        case Wild:
            ret = 1;
            point_add(P, P, target, param);
            break;
        case InverseWild:
            ret = -1;
            point_add(P, P, target_inv, param);
        }
        return ret;
    }
};

__global__ void kernel_create_kangaroos(cgbn_error_report_t *report, Kangaroo *kangaroos, ECPointCGBN *G_, ECPointCGBN *target_, ECPointCGBN *target_inv_, Curve *curve_, size_t count) {
    int32_t instance;
  
    // decode an instance number from the blockIdx and threadIdx
    instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
    if(instance>=count)
        return;
    
    context_t          bn_context(cgbn_report_monitor, report, instance);
    env_t              bn_env(bn_context);
    ECPointGPU G, target, target_inv, P0;
    CurveGPU curve;
    cgbn_load(bn_env, curve.p, &(curve_->p));
    cgbn_load(bn_env, curve.a, &(curve_->a));
    cgbn_load(bn_env, curve.b, &(curve_->b));
    cgbn_load(bn_env, curve._r, &(curve_->_r));
    cgbn_load(bn_env, curve._r2, &(curve_->_r2));
    cgbn_load(bn_env, curve._r3, &(curve_->_r3));
    curve.np0 = curve_->np0;

    load_point(bn_env, G, G_);
    load_point(bn_env, target, target_);
    load_point(bn_env, target_inv, target_inv_);

    env_t::cgbn_t offset, offset_sign;
    cgbn_load(bn_env, offset, &(kangaroos[instance].offset));
    auto herd = kangaroos[instance].herd;

    cgbn_shift_right(bn_env, offset_sign, offset, NBITS-1);
    if (cgbn_get_ui32(bn_env, offset_sign)>0) {
        cgbn_sub(bn_env, G.y, curve.p, G.y);
        cgbn_bitwise_complement(bn_env, offset, offset);
        cgbn_add_ui32(bn_env, offset, offset, 1);
    }
    ECPointExtGPU G_ext(bn_env, G, curve), P_ext;
    point_mul(bn_env, P_ext, G_ext, offset, curve);
    proj_point(bn_env, P0, P_ext, curve);
    if (herd == Wild) {
        point_add(bn_env, P0, P0, target, curve);
    } else if (herd == InverseWild) {
        point_add(bn_env, P0, P0, target_inv, curve);
    }
    save_point(bn_env, &(kangaroos[instance].P), P0);
}

__device__ __forceinline__ void save_output(env_t env, Kangaroo *output, uint32_t *output_idx, uint32_t *atom_pos, uint32_t idx, ECPointGPU &P, env_t::cgbn_t &offset, Herd herd) {
    bool is_main = (threadIdx.x % TPI == 0);
    uint32_t pos=4294967295U;
    if (is_main)
        pos = atomicAdd(atom_pos, 1);
    env_t::cgbn_t tmp;
    cgbn_set_ui32(env, tmp, pos);
    pos = cgbn_get_ui32(env, tmp);
    assert(pos < MAX_FOUND);
    save_point(env, &(output[pos].P), P);
    cgbn_store(env, &(output[pos].offset), offset);
    output[pos].herd = herd;
    output_idx[pos] = idx;
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

    ECPointGPU Ps[GRP_INV_SIZE];
    env_t::cgbn_t offset[GRP_INV_SIZE];
    Herd herd[GRP_INV_SIZE];
    JumpEntryGPU jumptable[64];
    for (int i = 0; i < GRP_INV_SIZE; i++) {
        load_point(bn_env, Ps[i], &(kangaroos[instance*GRP_INV_SIZE+i].P));
        cgbn_load(bn_env, offset[i], &(kangaroos[instance*GRP_INV_SIZE+i].offset));
        herd[i] = kangaroos[instance*GRP_INV_SIZE+i].herd;
    }
    for (int i = 0; i < 64; i++) {
        load_point(bn_env, jumptable[i].point, &(jumptable_[i].point));
        cgbn_load(bn_env, jumptable[i].offset, &(jumptable_[i].offset));
    }

    for (int nr = 0; nr < NB_RUN; nr++) {
        env_t::cgbn_t dy[GRP_INV_SIZE], dx[GRP_INV_SIZE];
        for (int i=0; i<GRP_INV_SIZE; i++) {
            auto &P = Ps[i];
            uint32_t jmp = cgbn_get_ui32(bn_env, P.y) >> (32-6);
            auto &Q = jumptable[jmp].point;
            mont_sub(bn_env, dy[i], P.y, Q.y, curve);
            mont_sub(bn_env, dx[i], P.x, Q.x, curve);
        }
        batch_inverse(bn_env, dy, dx, curve);
        env_t::cgbn_t t1, t2;
        for (int i=0; i<GRP_INV_SIZE; i++) {
            auto &P = Ps[i];
            uint32_t jmp = cgbn_get_ui32(bn_env, P.y) >> (32-6);
            auto &Q = jumptable[jmp].point;
            auto &lambda = dy[i];
            mont_sqr(bn_env, t1, lambda, curve);
            mont_sub(bn_env, t1, t1, P.x, curve);
            mont_sub(bn_env, t1, t1, Q.x, curve); // x3 = lambda^2 - x1 - x2
            mont_sub(bn_env, t2, P.x, t1, curve);
            cgbn_set(bn_env, P.x, t1);
            mont_mul(bn_env, t1, lambda, t2, curve);
            mont_sub(bn_env, P.y, t1, P.y, curve); // y3 = lambda*(x1-x3) - y1
            cgbn_add(bn_env, offset[i], offset[i], jumptable[jmp].offset);

            if ((cgbn_get_ui32(bn_env, P.x) & DP_MASK) == 0) {
                save_output(bn_env, output, output_idx, atom_pos, instance*GRP_INV_SIZE+i, P, offset[i], herd[i]);
            }
        }
    }
    
    for (int i = 0; i < GRP_INV_SIZE; i++) {
        save_point(bn_env, &(kangaroos[instance*GRP_INV_SIZE+i].P), Ps[i]);
        cgbn_store(bn_env, &(kangaroos[instance*GRP_INV_SIZE+i].offset), offset[i]);
    }
}

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

    mpz_set_str(max_offset, "ffffffffffffffff", 16);
    point_mul(target, G, 0x2ca6a807c356f637L, param);

    assert(mpz_cmp(max_offset, order) < 0);
}


void init_jumptable(JumpEntry *jumptable, size_t count)
{
    ECPoint P;
    mpz_t offsets[count];
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
        point_mul(P, G, offsets[i], param);
        jumptable[i].point.from_point(P);
        from_mpz_signed(jumptable[i].offset, offsets[i]);
        mpz_clear(offsets[i]);
    }
}

void shuffle_point(Kangaroo &ka, size_t range_size) {
    THREAD_LOCAL_MPZ_WORKSPACE(s);
    ECPoint P;
    switch (ka.herd)
    {
    case Tame:
        mpz_randombits(s, range_size, true);
        point_mul(P, G, s, param);
        break;
    case Wild:
        mpz_randombits(s, range_size-1, true);
        point_mul(P, G, s, param);
        point_add(P, P, target, param);
        break;
    case InverseWild:
        mpz_randombits(s, range_size-1, true);
        point_mul(P, G, s, param);
        point_add(P, P, target_inv, param);
    }
    ka.P.from_point(P);
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
    kernel_create_kangaroos<<<(count+(128/TPI-1))/(128/TPI), 128>>>(report, gpu_kangaroos, G_gpu, target_gpu, target_inv_gpu, gpuCurve, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    free(kangaroos);
    mpz_clear(s);
    return gpu_kangaroos;
}

void init() {
    ECPoint P0;
    ECPointCGBN G_cgbn, target_cgbn, target_inv_cgbn;
    assert(is_on_curve(G, param));
    point_mul(P0, G, order, param);
    assert(is_infinity(P0));

    mpz_div_2exp(mid, max_offset, 1); // mid = max_offset / 2
    point_mul(P0, G, mid, param);
    point_neg_(P0, param);
    point_add(target, P0, target, param); // target = target - mid * G
    point_set(target_inv, target);
    point_neg_(target_inv, param);

    G_cgbn.from_point(G);
    target_cgbn.from_point(target);
    target_inv_cgbn.from_point(target_inv);

    CUDA_CHECK(cudaMalloc(&G_gpu, sizeof(ECPointCGBN)));
    CUDA_CHECK(cudaMemcpy(G_gpu, &G_cgbn, sizeof(ECPointCGBN), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&target_gpu, sizeof(ECPointCGBN)));
    CUDA_CHECK(cudaMemcpy(target_gpu, &target_cgbn, sizeof(ECPointCGBN), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&target_inv_gpu, sizeof(ECPointCGBN)));
    CUDA_CHECK(cudaMemcpy(target_inv_gpu, &target_inv_cgbn, sizeof(ECPointCGBN), cudaMemcpyHostToDevice));
}


int main()
{
    ECPoint P0, P1;
    HashTable table;
    JumpEntry jumptable[64];
    cgbn_error_report_t *report;
    read_curve();

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    auto sm_count = prop.multiProcessorCount;
    auto num_kangaroo = sm_count * (128 / TPI) * GRP_INV_SIZE;
    printf("num_kangaroo: %d\n", num_kangaroo);
    init();
    init_jumptable(jumptable, 64);

    CUDA_CHECK(cudaMalloc(&jumptable_gpu, 64 * sizeof(JumpEntry)));
    CUDA_CHECK(cudaMemcpy(jumptable_gpu, jumptable, 64 * sizeof(JumpEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cgbn_error_report_alloc(&report));

    Curve *curve = new Curve(param);
    CUDA_CHECK(cudaMalloc(&gpuCurve, sizeof(Curve)));
    CUDA_CHECK(cudaMemcpy(gpuCurve, curve, sizeof(Curve), cudaMemcpyHostToDevice));

    size_t range_size = mpz_sizeinbase(mid, 2);
    Kangaroo *kangaroos = prepare_kangaroo(report, range_size, num_kangaroo);

    // Kangaroo *cpu_verify = (Kangaroo *)malloc(num_kangaroo * sizeof(Kangaroo));
    // CUDA_CHECK(cudaMemcpy(cpu_verify, kangaroos, num_kangaroo * sizeof(Kangaroo), cudaMemcpyDeviceToHost));
    // ECPoint A, B;
    // for (size_t i = 0; i < num_kangaroo; i++) {
    //     cpu_verify[i].P.to_point(A);
    //     uint128_t offset;
    //     store_low_256_bits(cpu_verify[i].offset, offset);
    //     table.compute_node(cpu_verify[i].herd, offset, B);
    //     if(!(A==B)){
    //         printf("%lu, herd %d\n", i, cpu_verify[i].herd);
    //         print_words(cpu_verify[i].offset._limbs, NBITS/32);
    //         print_point(A);
    //         print_point(B);
    //         exit(1);
    //     }
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

    bool find=false;
    size_t cnt=0;
    while(!find) {
        CUDA_CHECK(cudaMemset(gpu_atom_pos, 0, sizeof(uint32_t)));
        kernel_jump<<<sm_count, 128>>>(report, kangaroos, jumptable_gpu, gpuOutput, gpu_output_idx, gpu_atom_pos, gpuCurve, num_kangaroo/GRP_INV_SIZE);
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
    printf("cnt: %lu\n", cnt);
}

