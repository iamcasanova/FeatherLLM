#include "featherllm/storage/checkpoint_manifest.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

std::uint64_t shard_data_offset(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    return static_cast<std::uint64_t>(in.tellg());
}

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

void expect_rejected(const std::filesystem::path& index, const std::string& json) {
    {
        std::ofstream out(index);
        out << json;
    }
    bool rejected = false;
    try {
        (void)featherllm::storage::load_safetensors_index(index);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
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
    const auto data_offset = shard_data_offset(shard);
    assert(manifest.tensors.size() == 2);
    assert(manifest.tensors.at("a").shard == shard.filename().string());
    assert(manifest.tensors.at("a").data_offset == data_offset);
    assert(manifest.tensors.at("a").data_length == 3);
    assert(manifest.tensors.at("b").data_offset == data_offset + 3);
    assert(manifest.tensors.at("b").data_length == 4);

    {
        std::ofstream out(index);
        out << R"({"weight_map":{"a":"manifest_test-00001-of-00001.safetensors","b":"manifest_test-00001-of-00001.safetensors"}})";
    }
    const auto escaped_manifest = featherllm::storage::load_safetensors_index(index);
    assert(escaped_manifest.tensors.size() == 2);

    expect_rejected(index,
        R"({"weight_map":{"a":"manifest_test-00001-of-00001.safetensors","a":"manifest_test-00001-of-00001.safetensors"}})");
    expect_rejected(index,
        R"({"weight_map":{"a":"manifest_test-00001-of-00001.safetensors",}})");
    expect_rejected(index,
        R"({"weight_map":{"a":"manifest_test-00001-of-00001.safetensors" "b":"manifest_test-00001-of-00001.safetensors"}})");
    expect_rejected(index,
        R"({"weight_map":{"a":"manifest_test-00001-of-00001.safetensors\u0001"}})");
    expect_rejected(index,
        R"({"weight_map":{"a":"manifest_test-00001-of-00001.safetensors\uD800"}})");
    expect_rejected(index,
        R"({"weight_map":{"a":"../manifest_test-00001-of-00001.safetensors"}})");
    expect_rejected(index,
        R"({"weight_map":{"a":"/tmp/manifest_test-00001-of-00001.safetensors"}})");

    std::filesystem::remove(index);
    std::filesystem::remove(shard);
}
