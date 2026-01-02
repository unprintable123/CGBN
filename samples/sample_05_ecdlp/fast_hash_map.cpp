// g++ -O3 -march=native -mtune=native -std=c++20 -flto -fomit-frame-pointer -funroll-loops -fopenmp -o test fast_hash_map.cpp
#include <iostream>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <chrono>
#include <immintrin.h>

long get_resident_set_size() {
    std::ifstream stat_stream("/proc/self/status");
    std::string line;
    while (std::getline(stat_stream, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            return std::stol(line.substr(7));
        }
    }
    return 0;
}


class flat_hash_set {
    uint64_t _capacity;
    uint64_t *_buckets;

public:
    flat_hash_set(uint64_t capacity) {
        m1 = uint32_t(sqrt(capacity / 2) + 3) | 1;
        m2 = 2 * m1 + 1;
        _capacity = m1 * m2 + 100;
        _capacity += 16 - (_capacity % 16);
        if ((_capacity * sizeof(uint64_t) >> 30) > 32) {
            throw std::bad_alloc();
        }
        _buckets = new uint64_t[_capacity];

        memset(_buckets, 0xff, _capacity * sizeof(uint64_t));
    }

    ~flat_hash_set() {
        delete[] _buckets;
    }

    void insert(uint64_t index, uint64_t key) { 
        // assume index is uniformly distributed, index < m1*m2
        auto ptr = _buckets + index;
        while (*ptr != 0xffffffffffffffff) {
            ptr += 1;
        }
        *ptr = key;
    }

    bool find(uint64_t index, uint64_t key) {
        auto ptr = _buckets + index;
        while (*ptr != 0xffffffffffffffff) {
            if (*ptr == key) {
                return true;
            }
            ptr += 1;
        }
        return false;
    }

    uint32_t m1, m2;
};


class simd_flat_hash_set {
public:
    static constexpr uint8_t EMPTY = 0xFF;
    uint64_t _capacity;
    uint64_t *_buckets;
    uint8_t *_ctrl;
    uint32_t m1, m2;

    simd_flat_hash_set(uint64_t capacity) {
        m1 = uint32_t(sqrt((float)capacity / 2) + 3) | 1;
        m2 = 2 * m1 - 1;
        _capacity = m1 * m2 + 100;
        _capacity += 16 - (_capacity % 16);
        _buckets = (uint64_t*)aligned_alloc(64, _capacity * sizeof(uint64_t));
        _ctrl = (uint8_t*)aligned_alloc(64, _capacity);

        memset(_ctrl, EMPTY, _capacity);
    }

    ~simd_flat_hash_set() {
        free(_buckets);
        free(_ctrl);
    }

    void empty() {
        memset(_ctrl, EMPTY, _capacity);
    }

    void insert(uint64_t index, uint64_t key) { 
        // assume index is uniformly distributed, index < m1*m2
        while (true) {
            __m128i ctrl_group = _mm_loadu_si128(reinterpret_cast<const __m128i*>(_ctrl + index));
            uint32_t mask = _mm_movemask_epi8(_mm_cmpeq_epi8(ctrl_group, _mm_set1_epi8(EMPTY)));

            while (mask != 0) {
                int offset = index + __builtin_ctz(mask);

                uint8_t expected = EMPTY;
                if (__atomic_compare_exchange_n(_ctrl + offset, &expected, uint16_t(key) % 255, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                    // _buckets[offset] = key;
                    _mm_stream_si64((long long*)&_buckets[offset], key);
                    return;
                }

                mask = mask & (mask - 1);
            }
            index += 16;
        }
    }

    bool find(uint64_t index, uint64_t key) {
        uint8_t h2 = uint16_t(key) % 255;
        
        while (true) {
            __m128i ctrl_group = _mm_loadu_si128(reinterpret_cast<const __m128i*>(_ctrl + index));

            __m128i target = _mm_set1_epi8(static_cast<char>(h2));
            __m128i match_res = _mm_cmpeq_epi8(ctrl_group, target);
            uint32_t mask = _mm_movemask_epi8(match_res);

            while (mask != 0) {
                int bit_idx = __builtin_ctz(mask);
                if (_buckets[index + bit_idx] == key) return true;
                mask &= (mask - 1);
            }

            __m128i empty_res = _mm_cmpeq_epi8(ctrl_group, _mm_set1_epi8(static_cast<char>(EMPTY)));
            if (_mm_movemask_epi8(empty_res) != 0) return false;

            index += 16;
        }
    }
};

#define NBITS 30

uint64_t hash(uint64_t key) {
    key = (~key) + (key << 21);
    key = key * 0x120385ebca6bUL;
    key = key ^ (key >> 19);
    key = key * 0x7895deece66dUL;
    return key;
}

constexpr uint64_t size = uint64_t(1) << NBITS;
int main() {
#ifdef false
    {
    std::cout << "flat_hash_set" << std::endl;
    flat_hash_set map(size);

    auto start_insert = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < size; i++) {
        auto c = hash(i);
        auto id = hash(c) & (size - 1);
        if (c & 3)
            map.insert(id, c);
    }
    auto end_insert = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_insert - start_insert).count() << "ms" << std::endl;

    for (uint64_t i = 0; i < size; i++) {
        auto c = hash(i);
        auto id = hash(c) & (size - 1);
        if(map.find(id, c+1))std::cout << "error" << std::endl;
        // if (map.find(id, c) != ((c & 7) != 0)) {
        //     std::cout << "error" << std::endl;
        // }
    }
    auto end_find = std::chrono::high_resolution_clock::now();
    std::cout << "find time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_find - end_insert).count() << "ms" << std::endl;
    }
#endif

    {
    std::cout << "simd_flat_hash_set" << std::endl;
    simd_flat_hash_set map(size);

    auto start_insert = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (uint64_t i = 0; i < size; i++) {
        auto c = hash(i);
        auto id = hash(c) & (size - 1);
        if (c & 3)
            map.insert(id, c);
    }

    auto end_insert = std::chrono::high_resolution_clock::now();
    std::cout << "insert time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_insert - start_insert).count() << "ms" << std::endl;

    auto start_find = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (uint64_t i = 0; i < (4*size); i++) {
        auto c = hash(i);
        auto id = hash(~c) & (size - 1);
        if(map.find(id+1, c+1))std::cout << "error" << std::endl;
        // if (map.find(id, c) != ((c & 3) != 0 && i < size)) {
        //     std::cout << "error" << std::endl;
        // }
    }
    auto end_find = std::chrono::high_resolution_clock::now();
    std::cout << "find time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_find - start_find).count() << "ms" << std::endl;
    }
}






