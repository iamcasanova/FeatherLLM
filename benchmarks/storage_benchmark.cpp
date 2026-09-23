#include "featherllm/storage/bounded_file.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    const auto path = std::filesystem::temp_directory_path() / "featherllm_storage_benchmark.bin";
    constexpr std::size_t file_bytes = 32 * 1024 * 1024;
    constexpr std::size_t read_bytes = 1 * 1024 * 1024;
    constexpr std::size_t iterations = 64;

    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) return 2;
        std::vector<char> block(1024 * 1024, 0);
        for (std::size_t written = 0; written < file_bytes; written += block.size())
            out.write(block.data(), static_cast<std::streamsize>(block.size()));
    }

    featherllm::storage::BoundedFileReader reader(path, 4 * 1024 * 1024);
    std::vector<std::byte> buffer(read_bytes);
    std::uint64_t checksum = 0;

    const auto begin = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        const auto offset = static_cast<std::uint64_t>((i * read_bytes) % (file_bytes - read_bytes));
        reader.read(offset, buffer);
        for (const auto byte : buffer) checksum += std::to_integer<unsigned int>(byte);
    }
    const auto end = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(end - begin).count();
    const double mib = static_cast<double>(iterations * read_bytes) / (1024.0 * 1024.0);
    std::cout << "bytes_read=" << (iterations * read_bytes)
              << " seconds=" << seconds
              << " MiB_per_second=" << (mib / seconds)
              << " checksum=" << checksum << '\n';

    std::filesystem::remove(path);
    return 0;
}
