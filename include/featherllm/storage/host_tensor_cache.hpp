#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace featherllm::storage {

class HostTensorCache {
public:
    using Bytes = std::vector<std::byte>;

    struct Stats {
        std::uint64_t hits{0};
        std::uint64_t misses{0};
    };

    explicit HostTensorCache(std::size_t capacity_bytes);

    HostTensorCache(const HostTensorCache&) = delete;
    HostTensorCache& operator=(const HostTensorCache&) = delete;

    // Returns an immutable, zero-copy view of the cached bytes. Holding the
    // shared_ptr keeps the returned storage alive even if the entry is evicted.
    std::shared_ptr<const Bytes> get(std::string_view key);

    // Stores an immutable copy of the supplied tensor bytes. Returns false if
    // the tensor is larger than the cache capacity and therefore cannot fit.
    bool put(std::string key, Bytes data);

    void erase(std::string_view key);
    void clear();

    std::size_t capacity_bytes() const noexcept;
    std::size_t resident_bytes() const noexcept;
    std::size_t size() const noexcept;
    Stats stats() const noexcept;

private:
    struct Entry {
        std::shared_ptr<const Bytes> data;
        std::list<std::string>::iterator lru_position;
    };

    using Entries = std::unordered_map<std::string, Entry>;

    void erase_locked(Entries::iterator it);
    void evict_until_fit_locked(std::size_t incoming_bytes);
    void touch_locked(Entries::iterator it);

    const std::size_t capacity_bytes_;
    std::size_t resident_bytes_{0};
    std::uint64_t hits_{0};
    std::uint64_t misses_{0};
    std::list<std::string> lru_;
    Entries entries_;
    mutable std::mutex mutex_;
};

} // namespace featherllm::storage
