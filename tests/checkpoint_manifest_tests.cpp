#include "featherllm/storage/checkpoint_manifest.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

void write_shard(const std::filesystem::path& path) {
    const std::string header =
        "{\"a\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[0,3]},"
        "\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[3,7]}}";
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
    const auto shard = dir / "manifest_test-00001-of-00001.safetensors";
    const auto index = dir / "manifest_test.safetensors.index.json";
    write_shard(shard);

    {
        std::ofstream out(index);
        out << R"({"metadata":{},"weight_map":{"a":"manifest_test-00001-of-00001.safetensors","b":"manifest_test-00001-of-00001.safetensors"}})";
    }

    const auto manifest = featherllm::storage::load_safetensors_index(index);
    assert(manifest.tensors.size() == 2);
    assert(manifest.tensors.at("a").shard == shard.filename().string());
    assert(manifest.tensors.at("a").data_offset == 0);
    assert(manifest.tensors.at("a").data_length == 3);
    assert(manifest.tensors.at("b").data_offset == 3);
    assert(manifest.tensors.at("b").data_length == 4);

    std::filesystem::remove(index);
    std::filesystem::remove(shard);
}
