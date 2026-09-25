#include "featherllm/safetensors/tensor_reader.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>

namespace {

void write_shard(const std::filesystem::path& path) {
    const std::string header =
        "{\"a\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[0,3]},"
        "\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[3,7]}}";
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xff));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    const unsigned char bytes[7] = {1, 2, 3, 4, 5, 6, 7};
    out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

} // namespace

int main() {
    const auto dir = std::filesystem::current_path();
    const auto shard = dir / "lazy_tensor_view_test.safetensors";
    const auto index = dir / "lazy_tensor_view_test.safetensors.index.json";
    write_shard(shard);
    {
        std::ofstream out(index);
        out << R"({"weight_map":{"a":"lazy_tensor_view_test.safetensors","b":"lazy_tensor_view_test.safetensors"}})";
    }

    featherllm::safetensors::ShardedTensorReader reader(index, 4, 0);
    const auto view = reader.view_tensor("b");
    assert(view.is_mapped());
    assert(view.size() == 4);
    assert(view.data()[0] == std::byte{4});
    assert(view.data()[3] == std::byte{7});

    std::filesystem::remove(index);
    std::filesystem::remove(shard);
}
