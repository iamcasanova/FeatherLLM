#include "featherllm/storage/host_tensor_cache.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    constexpr std::size_t tensor_bytes = 1 * 1024 * 1024;
    constexpr std::size_t iterations = 256;

    featherllm::storage::HostTensorCache cache(tensor_bytes);
    if (cache.get("tensor")) return 2;
    const auto after_cold_miss = cache.stats();
    if (after_cold_miss.misses != 1 || after_cold_miss.hits != 0) return 3;

    std::vector<std::byte> tensor(tensor_bytes, std::byte{7});
    if (!cache.put("tensor", std::move(tensor))) return 4;

    std::uint64_t checksum = 0;
    const auto begin = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        const auto resident = cache.get("tensor");
        if (!resident || resident->empty()) return 5;
        checksum += std::to_integer<unsigned int>((*resident)[i % resident->size()]);
    }
    const auto end = std::chrono::steady_clock::now();

    const auto stats = cache.stats();
    if (stats.misses != 1 || stats.hits != iterations) return 6;

    const double seconds = std::chrono::duration<double>(end - begin).count();
    const double mib = static_cast<double>(iterations * tensor_bytes) / (1024.0 * 1024.0);
    const double hit_rate = static_cast<double>(stats.hits) /
                            static_cast<double>(stats.hits + stats.misses);
    std::cout << "cache_bytes_served=" << (iterations * tensor_bytes)
              << " iterations=" << iterations
              << " seconds=" << seconds
              << " logical_MiB_per_second=" << (mib / seconds)
              << " hits=" << stats.hits
              << " misses=" << stats.misses
              << " hit_rate=" << hit_rate
              << " resident_bytes=" << cache.resident_bytes()
              << " entries=" << cache.size()
              << " checksum=" << checksum << '\n';
    return 0;
}
