#pragma once

#include "featherllm/storage/checkpoint_manifest.hpp"
#include "featherllm/storage/bounded_file.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace featherllm::safetensors {

class ShardedTensorReader {
public:
    explicit ShardedTensorReader(std::filesystem::path index_path,
                                 std::size_t window_bytes = 4 * 1024 * 1024);

    [[nodiscard]] const storage::CheckpointManifest& manifest() const noexcept { return manifest_; }
    [[nodiscard]] std::vector<std::byte> read_tensor(const std::string& name);

private:
    std::filesystem::path index_path_;
    std::size_t window_bytes_;
    storage::CheckpointManifest manifest_;
    std::unordered_map<std::string, storage::BoundedFileReader> readers_;
};

} // namespace featherllm::safetensors
