#include <array>
#include <math.h>
#include <unordered_map>

#include "ecc.h"
#include "../../utility/cpu_support.h"

constexpr uint32_t DP_SIZE = 12;
constexpr uint32_t DP_MASK = (1 << DP_SIZE) - 1;

enum Herd {
    Tame = 0,
    Wild = 1,
    InverseWild = 2
};

struct Kangaroo
{
    ECPoint P;
    mpz_t offset;
    Herd herd;

    Kangaroo() {
        mpz_init(offset);
    }

    ~Kangaroo() {
        mpz_clear(offset);
    }
};

typedef std::array<uint32_t, 4> uint128_t;

struct Uint128Hasher {
    std::size_t operator()(const std::array<uint32_t, 4>& arr) const {
        std::size_t h = 0;
        for (uint32_t x : arr) {
            h ^= std::hash<uint32_t>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

void store_low_256_bits(mpz_t &src, uint128_t& dest, bool sgn=false) {
    THREAD_LOCAL_MPZ_WORKSPACE(low_bits);

    mpz_tdiv_r_2exp(low_bits, src, 128); 

    size_t count = 0;
    dest.fill(0);

    mpz_export(dest.data(), &count, -1, sizeof(uint32_t), 0, 0, low_bits);
    if (sgn && (mpz_sgn(src) < 0)) {
        dest[3] |= (1<<31);
    }
}

void load_uint128(uint128_t& dest, mpz_t &src, bool use_sgn=false) {
    bool sgn = dest[3] >> 31;
    if (use_sgn) {
        dest[3] &= ~(1<<31);
    }
    mpz_import(src, 4, -1, sizeof(uint32_t), 0, 0, dest.data());
    if (use_sgn && sgn) {
        mpz_neg(src, src);
    }
}

struct HashEntry
{
    uint128_t offset;
    Herd herd;
};

struct JumpEntry
{
    ECPoint P;
    mpz_t offset;

    JumpEntry() {
        mpz_init(offset);
    }

    ~JumpEntry() {
        mpz_clear(offset);
    }
};


ECParameters param;
ECPoint G, target, target_inv;
mpz_t order, max_offset, mid;

struct HashTable
{
    std::unordered_map<uint128_t, HashEntry, Uint128Hasher> entries;
    HashTable() {
        entries.reserve(1000000);
    }

    int addPoint(Kangaroo &point) {
        uint128_t position, offset;
        store_low_256_bits(point.P.x, position);
        store_low_256_bits(point.offset, offset, true);
        
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
                mpz_clears(o1, o2, NULL);
                return -1;
            }
            mpz_set_si(o2, -f1);
            mpz_invert(o2, o2, order);
            mpz_mul(o1, o1, o2);
            mpz_add(o1, o1, mid);
            mpz_mod(o1, o1, order);
            
            printf("collision detected\n");
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

    mpz_set_str(max_offset, "fffffffffff", 16);
    point_mul(target, G, 0x87c356f637L, param);

    assert(mpz_cmp(max_offset, order) < 0);
}

void init_jumptable(JumpEntry *jumptable, size_t count)
{
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
                mpz_randombits(jumptable[i].offset, jump_bits);
            } while (mpz_get_d(jumptable[i].offset) > max_range);
            dist_avg += mpz_get_d(jumptable[i].offset);
        }
        dist_avg /= count;
        if (dist_avg <= max_avg && dist_avg >= min_avg) {
            break;
        }
    }
    printf("dist_avg: %f\n", log2(dist_avg));
    for (size_t i = 0; i < count; i++) {
        point_mul(jumptable[i].P, G, jumptable[i].offset, param);
    }
}

void shuffle_point(Kangaroo &ka, size_t range_size) {
    THREAD_LOCAL_MPZ_WORKSPACE(s);
    switch (ka.herd)
    {
    case Tame:
        mpz_randombits(s, range_size, true);
        point_mul(ka.P, G, s, param);
        mpz_set(ka.offset, s);
        break;
    case Wild:
        mpz_randombits(s, range_size-1, true);
        point_mul(ka.P, G, s, param);
        point_add(ka.P, ka.P, target, param);
        mpz_set(ka.offset, s);
        break;
    case InverseWild:
        mpz_randombits(s, range_size-1, true);
        point_mul(ka.P, G, s, param);
        point_add(ka.P, ka.P, target_inv, param);
        mpz_set(ka.offset, s);
    }
}

int main()
{
    ECPoint P0, P1;
    HashTable table;
    JumpEntry jumptable[64];
    read_curve();
    init_jumptable(jumptable, 64);

    assert(is_on_curve(G, param));
    point_mul(P0, G, order, param);
    assert(is_infinity(P0));

    mpz_div_2exp(mid, max_offset, 1); // mid = max_offset / 2
    point_mul(P0, G, mid, param);
    point_neg_(P0, param);
    point_add(target, P0, target, param); // target = target - mid * G
    point_set(target_inv, target);
    point_neg_(target_inv, param);

    Kangaroo *kangaroos = new Kangaroo[400];
    size_t range_size = mpz_sizeinbase(mid, 2);
    
    for (int i = 0; i < 400; i++) {
        auto &ka = kangaroos[i];
        ka.herd = (Herd)(i % 3);
        shuffle_point(ka, range_size);
    }
    
    mpz_t s;
    mpz_init(s);
    bool find=false;
    size_t cnt=0;
    while(!find) {
        cnt+=400;
        for (int i = 0; i < 400; i++) {
            auto &ka = kangaroos[i];
            auto c = mpz_get_ui(ka.P.x);
            point_add(ka.P, ka.P, jumptable[c&63].P, param);
            mpz_add(ka.offset, ka.offset, jumptable[c&63].offset);
            mpz_tdiv_q_2exp(s, ka.P.x, 144);
            c = mpz_get_ui(s);
            if ((c & DP_MASK) == 0) {
                int ret = table.addPoint(ka);
                if (ret == 1) {
                    find = true;
                    break;
                } else if (ret != 0) {
                    shuffle_point(ka, range_size);
                }
            }
        }
    }
    printf("%f\n", log2(cnt));
    mpz_clear(s);
}

