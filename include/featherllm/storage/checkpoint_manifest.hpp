#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>

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

} // namespace featherllm::storage
