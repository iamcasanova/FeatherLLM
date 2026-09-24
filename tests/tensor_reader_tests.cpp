#include "featherllm/safetensors/tensor_reader.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

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
    const auto shard = dir / "tensor_reader_test.safetensors";
    const auto index = dir / "tensor_reader_test.safetensors.index.json";
    write_shard(shard);
    {
        std::ofstream out(index);
        out << R"({"weight_map":{"a":"tensor_reader_test.safetensors","b":"tensor_reader_test.safetensors"}})";
    }

    featherllm::safetensors::ShardedTensorReader reader(index, 4);
    const auto a = reader.read_tensor("a");
    const auto b = reader.read_tensor("b");
    const std::vector<std::byte> expected_a{std::byte{1}, std::byte{2}, std::byte{3}};
    const std::vector<std::byte> expected_b{std::byte{4}, std::byte{5}, std::byte{6}, std::byte{7}};
    assert(a == expected_a);
    assert(b == expected_b);

    bool rejected = false;
    try {
        (void)reader.read_tensor("missing");
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);

    std::filesystem::remove(index);
    std::filesystem::remove(shard);
}
