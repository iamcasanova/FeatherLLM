#pragma once

#include "featherllm/storage/checkpoint_manifest.hpp"
#include "featherllm/storage/bounded_file.hpp"
#include "featherllm/storage/host_tensor_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace featherllm::safetensors {

class ShardedTensorReader {
public:
    explicit ShardedTensorReader(std::filesystem::path index_path,
                                 std::size_t window_bytes = 4 * 1024 * 1024,
                                 std::size_t host_cache_capacity_bytes = 0);

    [[nodiscard]] const storage::CheckpointManifest& manifest() const noexcept { return manifest_; }
    [[nodiscard]] std::vector<std::byte> read_tensor(const std::string& name);
    [[nodiscard]] std::vector<std::byte> read_tensor_range(
        const std::string& name,
        std::uint64_t relative_offset,
        std::size_t max_bytes);
    [[nodiscard]] std::shared_ptr<const storage::HostTensorCache::Bytes>
    cached_tensor(const std::string& name);

private:
    std::filesystem::path index_path_;
    std::size_t window_bytes_;
    storage::HostTensorCache cache_;
    storage::CheckpointManifest manifest_;
    std::unordered_map<std::string, storage::BoundedFileReader> readers_;
};

} // namespace featherllm::safetensors
