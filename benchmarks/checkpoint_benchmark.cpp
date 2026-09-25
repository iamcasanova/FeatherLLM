#include "featherllm/safetensors/tensor_reader.hpp"
#include "featherllm/storage/checkpoint_manifest.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_shard(const std::filesystem::path& path, std::size_t tensor_bytes) {
    const std::string header =
        "{\"weights\":{\"dtype\":\"U8\",\"shape\":[" +
        std::to_string(tensor_bytes) +
        "],\"data_offsets\":[0," +
        std::to_string(tensor_bytes) +
        "]}}";

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create benchmark shard");

    const std::uint64_t header_size = header.size();
    for (int i = 0; i < 8; ++i)
        out.put(static_cast<char>((header_size >> (8 * i)) & 0xff));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    std::vector<char> block(1024 * 1024);
    for (std::size_t offset = 0; offset < tensor_bytes; offset += block.size()) {
        const auto remaining = tensor_bytes - offset;
        const auto count = std::min(block.size(), remaining);
        for (std::size_t i = 0; i < count; ++i)
            block[i] = static_cast<char>((offset + i) & 0xff);
        out.write(block.data(), static_cast<std::streamsize>(count));
    }
}

} // namespace

int main() {
    const auto base = std::filesystem::temp_directory_path();
    const auto shard = base / "featherllm_checkpoint_benchmark.safetensors";
    const auto index = base / "featherllm_checkpoint_benchmark.safetensors.index.json";

    constexpr std::size_t tensor_bytes = 4 * 1024 * 1024;
    constexpr std::size_t read_bytes = 256 * 1024;
    constexpr std::size_t iterations = 64;

    try {
        write_shard(shard, tensor_bytes);
        {
            std::ofstream out(index, std::ios::trunc);
            if (!out) throw std::runtime_error("cannot create benchmark index");
            out << R"({"weight_map":{"weights":"featherllm_checkpoint_benchmark.safetensors"}})";
        }

        const auto begin = std::chrono::steady_clock::now();
        std::uint64_t checksum = 0;
        for (std::size_t i = 0; i < iterations; ++i) {
            const auto bytes = featherllm::storage::read_tensor_bytes(
                index, "weights", (i * read_bytes) % (tensor_bytes - read_bytes), read_bytes);
            for (const auto byte : bytes)
                checksum += std::to_integer<unsigned int>(byte);
        }
        const auto end = std::chrono::steady_clock::now();

        featherllm::safetensors::ShardedTensorReader reader(index, 4 * 1024 * 1024);
        const auto reusable_begin = std::chrono::steady_clock::now();
        std::uint64_t reusable_checksum = 0;
        for (std::size_t i = 0; i < iterations; ++i) {
            const auto bytes = reader.read_tensor_range(
                "weights", (i * read_bytes) % (tensor_bytes - read_bytes), read_bytes);
            for (const auto byte : bytes)
                reusable_checksum += std::to_integer<unsigned int>(byte);
        }
        const auto reusable_end = std::chrono::steady_clock::now();

        const double seconds = std::chrono::duration<double>(end - begin).count();
        const double reusable_seconds =
            std::chrono::duration<double>(reusable_end - reusable_begin).count();
        const double mib = static_cast<double>(iterations * read_bytes) / (1024.0 * 1024.0);

        if (reusable_checksum != checksum)
            throw std::runtime_error("reusable reader checksum mismatch");

        std::cout << "checkpoint_tensor_bytes=" << tensor_bytes
                  << " bytes_read=" << (iterations * read_bytes)
                  << " iterations=" << iterations
                  << " seconds=" << seconds
                  << " MiB_per_second=" << (mib / seconds)
                  << " reusable_reader_seconds=" << reusable_seconds
                  << " reusable_reader_MiB_per_second=" << (mib / reusable_seconds)
                  << " checksum=" << checksum
                  << " reusable_checksum=" << reusable_checksum << '\n';

        std::filesystem::remove(index);
        std::filesystem::remove(shard);
        return 0;
    } catch (...) {
        std::filesystem::remove(index);
        std::filesystem::remove(shard);
        throw;
    }
}
