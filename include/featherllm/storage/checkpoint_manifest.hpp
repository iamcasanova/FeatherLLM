#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace featherllm::storage {

struct ShardTensorLocation {
  std::string shard;
  std::uint64_t data_offset{};
  std::uint64_t data_length{};
};

struct CheckpointManifest {
  std::unordered_map<std::string, ShardTensorLocation> tensors;
};

// Builds a tensor-to-shard/range manifest from a Hugging Face-style
// safetensors index JSON and the referenced shard files.
CheckpointManifest load_safetensors_index(const std::filesystem::path& index_path);

// Reads at most max_bytes from a tensor's data range, starting at relative_offset.
// The returned bytes are bounded by the tensor's declared range and max_bytes.
std::vector<std::byte> read_tensor_bytes(const std::filesystem::path& index_path,
                                         const std::string& tensor_name,
                                         std::uint64_t relative_offset = 0,
                                         std::size_t max_bytes = 4 * 1024 * 1024);

} // namespace featherllm::storage
