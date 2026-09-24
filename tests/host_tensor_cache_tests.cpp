#include "featherllm/storage/host_tensor_cache.hpp"

#include <cassert>
#include <cstddef>
#include <initializer_list>

namespace {

featherllm::storage::HostTensorCache::Bytes bytes(std::initializer_list<unsigned char> values) {
    featherllm::storage::HostTensorCache::Bytes result;
    result.reserve(values.size());
    for (const auto value : values) result.push_back(static_cast<std::byte>(value));
    return result;
}

} // namespace

int main() {
    featherllm::storage::HostTensorCache cache(8);
    assert(cache.capacity_bytes() == 8);
    assert(cache.size() == 0);
    assert(cache.resident_bytes() == 0);
    assert((cache.stats().hits == 0));
    assert((cache.stats().misses == 0));

    assert(cache.put("a", bytes({1, 2, 3, 4})));
    assert(cache.put("b", bytes({5, 6, 7, 8})));
    assert(cache.resident_bytes() == 8);

    const auto a = cache.get("a");
    assert(a);
    assert(a->size() == 4);
    assert((*a)[0] == std::byte{1});

    assert(cache.put("c", bytes({9, 10, 11, 12})));
    assert(!cache.get("b"));
    assert(cache.get("a"));
    assert(cache.get("c"));
    assert(cache.resident_bytes() == 8);

    const auto retained = cache.get("a");
    assert(retained);
    cache.erase("a");
    assert(!cache.get("a"));
    assert(retained->size() == 4);
    assert((*retained)[0] == std::byte{1});

    const auto stats = cache.stats();
    assert(stats.hits == 4);
    assert(stats.misses == 2);

    assert(!cache.put("too_large", bytes({1, 2, 3, 4, 5, 6, 7, 8, 9})));
    assert(cache.resident_bytes() == 4);

    cache.clear();
    assert(cache.size() == 0);
    assert(cache.resident_bytes() == 0);
    return 0;
}
