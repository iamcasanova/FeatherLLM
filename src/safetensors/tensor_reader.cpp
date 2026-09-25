#include "featherllm/safetensors/tensor_reader.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace featherllm::safetensors {

ShardedTensorReader::ShardedTensorReader(std::filesystem::path index_path,
                                         std::size_t window_bytes,
                                         std::size_t host_cache_capacity_bytes)
    : index_path_(std::move(index_path)),
      window_bytes_(window_bytes),
      cache_(host_cache_capacity_bytes),
      manifest_(storage::load_safetensors_index(index_path_)) {
    if (window_bytes_ == 0) throw std::invalid_argument("window size must be non-zero");
}

std::shared_ptr<const storage::HostTensorCache::Bytes>
ShardedTensorReader::cached_tensor(const std::string& name) {
    return cache_.get(name);
}

const storage::ShardTensorLocation& ShardedTensorReader::locate(
    const std::string& name) const {
    const auto it = manifest_.tensors.find(name);
    if (it == manifest_.tensors.end()) {
        throw std::out_of_range("tensor not found: " + name);
    }
    return it->second;
}

storage::BoundedFileReader& ShardedTensorReader::reader_for(const std::string& shard) {
    auto reader_it = readers_.find(shard);
    if (reader_it != readers_.end()) return reader_it->second;

    const auto path = index_path_.parent_path() / shard;
    auto [inserted, ok] = readers_.try_emplace(shard, path, window_bytes_);
    (void)ok;
    return inserted->second;
}

storage::MappedFileReader& ShardedTensorReader::mapped_reader_for(
    const std::string& shard) {
    auto reader_it = mapped_readers_.find(shard);
    if (reader_it != mapped_readers_.end()) return reader_it->second;

    const auto path = index_path_.parent_path() / shard;
    auto [inserted, ok] = mapped_readers_.try_emplace(shard, path);
    (void)ok;
    return inserted->second;
}

std::vector<std::byte> ShardedTensorReader::read_tensor(const std::string& name) {
    const auto cached = cache_.get(name);
    if (cached) return *cached;

    const auto& location = locate(name);
    if (location.data_length > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("tensor is too large for this process");
    }

    auto& reader = reader_for(location.shard);
    std::vector<std::byte> data(static_cast<std::size_t>(location.data_length));
    reader.read(location.data_offset, data);
    if (cache_.capacity_bytes() != 0) cache_.put(name, data);
    return data;
}

std::vector<std::byte> ShardedTensorReader::read_tensor_range(
    const std::string& name,
    const std::uint64_t relative_offset,
    const std::size_t max_bytes) {
    const auto& location = locate(name);
    if (relative_offset > location.data_length) {
        throw std::out_of_range(
            "tensor read offset exceeds tensor range: " + name);
    }

    const auto remaining = location.data_length - relative_offset;
    const auto requested = std::min<std::uint64_t>(
        remaining, static_cast<std::uint64_t>(max_bytes));

    auto& reader = reader_for(location.shard);
    std::vector<std::byte> data(static_cast<std::size_t>(requested));
    if (!data.empty()) {
        reader.read(location.data_offset + relative_offset, data);
    }
    return data;
}

TensorView ShardedTensorReader::view_tensor(const std::string& name) {
    const auto cached = cache_.get(name);
    if (cached) return TensorView(cached);

    const auto& location = locate(name);
    if (location.data_length > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("tensor is too large for this process");
    }

    auto& reader = mapped_reader_for(location.shard);
    return TensorView(reader.map(
        location.data_offset,
        static_cast<std::size_t>(location.data_length)));
}

} // namespace featherllm::safetensors
