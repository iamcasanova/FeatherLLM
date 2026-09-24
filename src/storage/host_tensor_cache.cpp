#include "featherllm/storage/host_tensor_cache.hpp"

#include <utility>

namespace featherllm::storage {

HostTensorCache::HostTensorCache(const std::size_t capacity_bytes)
    : capacity_bytes_(capacity_bytes) {}

std::shared_ptr<const HostTensorCache::Bytes> HostTensorCache::get(const std::string_view key) {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(std::string(key));
    if (it == entries_.end()) {
        ++misses_;
        return {};
    }
    ++hits_;
    touch_locked(it);
    return it->second.data;
}

bool HostTensorCache::put(std::string key, Bytes data) {
    const std::size_t incoming_bytes = data.size();
    std::lock_guard lock(mutex_);

    if (incoming_bytes > capacity_bytes_) {
        const auto existing = entries_.find(key);
        if (existing != entries_.end()) erase_locked(existing);
        return false;
    }

    const auto existing = entries_.find(key);
    if (existing != entries_.end()) erase_locked(existing);

    evict_until_fit_locked(incoming_bytes);
    lru_.push_front(key);
    const auto lru_position = lru_.begin();
    auto data_ptr = std::make_shared<const Bytes>(std::move(data));
    entries_.emplace(*lru_position, Entry{std::move(data_ptr), lru_position});
    resident_bytes_ += incoming_bytes;
    return true;
}

void HostTensorCache::erase(const std::string_view key) {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(std::string(key));
    if (it != entries_.end()) erase_locked(it);
}

void HostTensorCache::clear() {
    std::lock_guard lock(mutex_);
    entries_.clear();
    lru_.clear();
    resident_bytes_ = 0;
}

std::size_t HostTensorCache::capacity_bytes() const noexcept {
    return capacity_bytes_;
}

std::size_t HostTensorCache::resident_bytes() const noexcept {
    std::lock_guard lock(mutex_);
    return resident_bytes_;
}

std::size_t HostTensorCache::size() const noexcept {
    std::lock_guard lock(mutex_);
    return entries_.size();
}

HostTensorCache::Stats HostTensorCache::stats() const noexcept {
    std::lock_guard lock(mutex_);
    return Stats{hits_, misses_};
}

void HostTensorCache::erase_locked(const Entries::iterator it) {
    resident_bytes_ -= it->second.data->size();
    lru_.erase(it->second.lru_position);
    entries_.erase(it);
}

void HostTensorCache::evict_until_fit_locked(const std::size_t incoming_bytes) {
    // Avoid unsigned wraparound in resident_bytes_ + incoming_bytes.
    while (resident_bytes_ > capacity_bytes_ - incoming_bytes && !lru_.empty()) {
        const auto key = lru_.back();
        const auto it = entries_.find(key);
        if (it != entries_.end()) erase_locked(it);
        else lru_.pop_back();
    }
}

void HostTensorCache::touch_locked(const Entries::iterator it) {
    lru_.splice(lru_.begin(), lru_, it->second.lru_position);
    it->second.lru_position = lru_.begin();
}

} // namespace featherllm::storage
