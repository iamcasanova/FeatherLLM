#pragma once

#include "featherllm/storage/checkpoint_manifest.hpp"
#include "featherllm/storage/bounded_file.hpp"
#include "featherllm/storage/host_tensor_cache.hpp"
#include "featherllm/storage/mapped_file.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace featherllm::safetensors {

class TensorView {
public:
    using Bytes = storage::HostTensorCache::Bytes;

    [[nodiscard]] const std::byte* data() const noexcept {
        if (cached_) return cached_->data();
        return mapped_ ? mapped_->data() : nullptr;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        if (cached_) return cached_->size();
        return mapped_ ? mapped_->size() : 0;
    }

    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool is_mapped() const noexcept { return static_cast<bool>(mapped_); }

private:
    friend class ShardedTensorReader;

    explicit TensorView(std::shared_ptr<const Bytes> cached)
        : cached_(std::move(cached)) {}

    explicit TensorView(std::shared_ptr<const storage::MappedFileRegion> mapped)
        : mapped_(std::move(mapped)) {}

    std::shared_ptr<const Bytes> cached_;
    std::shared_ptr<const storage::MappedFileRegion> mapped_;
};

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

    // Returns a zero-copy tensor view. Cached tensors use the existing host
    // residency cache; uncached tensors are backed by an OS file mapping and
    // are paged in lazily by the operating system.
    [[nodiscard]] TensorView view_tensor(const std::string& name);

private:
    [[nodiscard]] const storage::ShardTensorLocation& locate(
        const std::string& name) const;
    storage::BoundedFileReader& reader_for(const std::string& shard);
    storage::MappedFileReader& mapped_reader_for(const std::string& shard);

    std::filesystem::path index_path_;
    std::size_t window_bytes_;
    storage::HostTensorCache cache_;
    storage::CheckpointManifest manifest_;
    std::unordered_map<std::string, storage::BoundedFileReader> readers_;
    std::unordered_map<std::string, storage::MappedFileReader> mapped_readers_;
};

} // namespace featherllm::safetensors
