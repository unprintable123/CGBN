#include <iostream>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <unistd.h>

// 获取当前进程的物理内存占用 (KB)
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

uint64_t hash(uint64_t key) {
    key = (~key) + (key << 21);
    key = key * 0x120385ebca6bUL;
    key = key ^ (key >> 19);
    key = key * 0x7895deece66dUL;
    return key;
}
int main() {
    size_t n = 1 << 26; // 2^24
    
    long mem_before = get_resident_set_size();
    
    // 使用自定义哈希
    std::unordered_map<uint64_t, uint64_t> map;
    
    // 建议预留空间减少 rehash 干扰，或者不预留以观察默认行为
    map.reserve(n);

    for (uint64_t i = 0; i < n; ++i) {
        auto c = hash(i);
        auto id = hash(c);
        if (c & 3)
        map[id] = c;
    }

    long mem_after = get_resident_set_size();
    
    double diff_mb = (mem_after - mem_before) / 1024.0;
    
    std::cout << "元素数量: " << n << std::endl;
    std::cout << "内存增量: " << diff_mb << " MB" << std::endl;
    std::cout << "平均每个 Key 占用: " << (diff_mb * 1024 * 1024 / n) << " 字节" << std::endl;

// #pragma omp parallel for
    for (uint64_t i = n; i < 2*n; ++i) {
        auto c = hash(i);
        auto id = hash(c);
        if (map.contains(id)) {
            std::cout << "contains: " << i << std::endl;
        }
    }

    return 0;
}