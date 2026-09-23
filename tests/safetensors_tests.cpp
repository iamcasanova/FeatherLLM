#include "featherllm/safetensors.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>

int main() {
    const char* path = "featherllm_test.safetensors";
    const std::string header =
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8*i)) & 0xff));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    const float values[2] = {1.0f, 2.0f};
    out.write(reinterpret_cast<const char*>(values), sizeof(values));
    out.close();

    featherllm::safetensors::Reader reader(path);
    reader.open();
    assert(reader.tensors().size() == 1);
    assert(reader.tensors().at("x").dtype == "F32");
    assert(reader.tensors().at("x").shape.size() == 1);
    assert(reader.tensors().at("x").shape[0] == 2);
    const auto bytes = reader.read_tensor("x");
    assert(bytes.size() == sizeof(values));

    std::remove(path);
    std::cout << "safetensors tests passed\n";
}
