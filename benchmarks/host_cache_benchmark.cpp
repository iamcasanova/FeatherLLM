#include "featherllm/safetensors/tensor_reader.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void write_fixture(const std::filesystem::path& shard,
                   const std::filesystem::path& index,
                   std::size_t tensor_bytes) {
    const std::string header =
        "{\"tensor\":{\"dtype\":\"U8\",\"shape\":[" +
        std::to_string(tensor_bytes) +
        "],\"data_offsets\":[0," + std::to_string(tensor_bytes) + "]}}";

    std::ofstream out(shard, std::ios::binary | std::ios::trunc);
    const std::uint64_t header_size = header.size();
    for (int i = 0; i < 8; ++i) {
        out.put(static_cast<char>((header_size >> (8 * i)) & 0xff));
    }
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    std::vector<char> payload(1024 * 1024, static_cast<char>(7));
    std::size_t remaining = tensor_bytes;
    while (remaining != 0) {
        const std::size_t chunk = std::min(remaining, payload.size());
        out.write(payload.data(), static_cast<std::streamsize>(chunk));
        remaining -= chunk;
    }

    std::ofstream manifest(index, std::ios::trunc);
    manifest << "{\"weight_map\":{\"tensor\":\"" << shard.filename().string()
             << "\"}}";
}

std::uint64_t checksum(const std::vector<std::byte>& bytes) {
    std::uint64_t value = 0;
    for (const auto byte : bytes) {
        value += std::to_integer<unsigned int>(byte);
    }
    return value;
}

} // namespace

int main() {
    constexpr std::size_t tensor_bytes = 4 * 1024 * 1024;
    constexpr std::size_t hot_iterations = 32;

    const auto dir = std::filesystem::current_path();
    const auto shard = dir / "host_cache_benchmark.safetensors";
    const auto index = dir / "host_cache_benchmark.safetensors.index.json";
    write_fixture(shard, index, tensor_bytes);

    int result = 0;
    {
        featherllm::safetensors::ShardedTensorReader reader(index, 4 * 1024 * 1024,
                                                            tensor_bytes);

        const auto cold_begin = std::chrono::steady_clock::now();
        const auto cold = reader.read_tensor("tensor");
        const auto cold_end = std::chrono::steady_clock::now();
        if (cold.size() != tensor_bytes || checksum(cold) == 0) {
            result = 2;
        } else {
            const auto resident = reader.cached_tensor("tensor");
            if (!resident || resident->size() != tensor_bytes) {
                result = 3;
            } else {
                std::uint64_t hot_checksum = 0;
                const auto hot_begin = std::chrono::steady_clock::now();
                for (std::size_t i = 0; i < hot_iterations; ++i) {
                    const auto hot = reader.cached_tensor("tensor");
                    if (!hot || hot->empty()) {
                        result = 4;
                        break;
                    }
                    hot_checksum +=
                        std::to_integer<unsigned int>((*hot)[i % hot->size()]);
                }
                const auto hot_end = std::chrono::steady_clock::now();

                const double cold_seconds =
                    std::chrono::duration<double>(cold_end - cold_begin).count();
                const double hot_seconds =
                    std::chrono::duration<double>(hot_end - hot_begin).count();
                const double cold_mib =
                    static_cast<double>(tensor_bytes) / (1024.0 * 1024.0);
                const double hot_mib =
                    static_cast<double>(hot_iterations * tensor_bytes) /
                    (1024.0 * 1024.0);

                std::cout << "tensor_bytes=" << tensor_bytes
                          << " cold_seconds=" << cold_seconds
                          << " cold_MiB_per_second=" << (cold_mib / cold_seconds)
                          << " hot_iterations=" << hot_iterations
                          << " hot_seconds=" << hot_seconds
                          << " hot_logical_MiB_per_second="
                          << (hot_mib / hot_seconds)
                          << " hot_checksum=" << hot_checksum << '\n';
            }
        }
    }

    std::filesystem::remove(index);
    std::filesystem::remove(shard);
    return result;
}
