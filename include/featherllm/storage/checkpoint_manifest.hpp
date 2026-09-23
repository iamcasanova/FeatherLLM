#pragma once

#include <cstdint>
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

} // namespace featherllm::storage
