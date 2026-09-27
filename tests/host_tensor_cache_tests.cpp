#include "featherllm/storage/host_tensor_cache.hpp"

#include <cstddef>
#include <cstdlib>
#include <initializer_list>

namespace {

void require(bool condition) {
    if (!condition) std::abort();
}

featherllm::storage::HostTensorCache::Bytes bytes(std::initializer_list<unsigned char> values) {
    featherllm::storage::HostTensorCache::Bytes result;
    result.reserve(values.size());
    for (const auto value : values) result.push_back(static_cast<std::byte>(value));
    return result;
}

} // namespace

int main() {
    featherllm::storage::HostTensorCache cache(8);
    require(cache.capacity_bytes() == 8);
    require(cache.size() == 0);
    require(cache.resident_bytes() == 0);
    require((cache.stats().hits == 0));
    require((cache.stats().misses == 0));

    require(cache.put("a", bytes({1, 2, 3, 4})));
    require(cache.put("b", bytes({5, 6, 7, 8})));
    require(cache.resident_bytes() == 8);

    const auto a = cache.get("a");
    require(a);
    require(a->size() == 4);
    require((*a)[0] == std::byte{1});

    require(cache.put("c", bytes({9, 10, 11, 12})));
    require(!cache.get("b"));
    require(cache.get("a"));
    require(cache.get("c"));
    require(cache.resident_bytes() == 8);

    const auto retained = cache.get("a");
    require(retained);
    cache.erase("a");
    require(!cache.get("a"));
    require(retained->size() == 4);
    require((*retained)[0] == std::byte{1});

    const auto stats = cache.stats();
    require(stats.hits == 4);
    require(stats.misses == 2);

    require(!cache.put("too_large", bytes({1, 2, 3, 4, 5, 6, 7, 8, 9})));
    require(cache.resident_bytes() == 4);

    require(cache.put("keep", bytes({1, 2, 3, 4})));
    require(!cache.put("keep", bytes({1, 2, 3, 4, 5, 6, 7, 8, 9})));
    const auto kept = cache.get("keep");
    require(kept);
    require(kept->size() == 4);
    require((*kept)[0] == std::byte{1});

    featherllm::storage::HostTensorCache disabled(0);
    require(!disabled.put("empty", {}));
    require(disabled.size() == 0);
    require(disabled.resident_bytes() == 0);

    cache.clear();
    require(cache.size() == 0);
    require(cache.resident_bytes() == 0);
    return 0;
}
