#include "featherllm/safetensors.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    const char* path = "featherllm_test.safetensors";
    const std::string header =
        "{\"__metadata__\":{\"format\":\"pt\",\"note\":\"test\"},"
        "\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
        "\"y\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[8,11]}}";

    std::ofstream out(path, std::ios::binary);
    const std::uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xff));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    const float values[2] = {1.0f, 2.0f};
    out.write(reinterpret_cast<const char*>(values), sizeof(values));
    const unsigned char tail[3] = {7, 8, 9};
    out.write(reinterpret_cast<const char*>(tail), sizeof(tail));
    out.close();

    featherllm::safetensors::Reader reader(path);
    reader.open();
    assert(reader.tensors().size() == 2);
    assert(reader.tensors().at("x").dtype == "F32");
    assert(reader.tensors().at("x").shape.size() == 1);
    assert(reader.tensors().at("x").shape[0] == 2);
    assert(reader.tensors().at("x").data_begin == 0);
    assert(reader.tensors().at("x").data_end == 8);

    const auto x = reader.read_tensor("x");
    assert(x.size() == sizeof(values));
    const auto y = reader.read_tensor("y");
    assert(y.size() == sizeof(tail));
    assert(std::to_integer<unsigned char>(y[0]) == 7);
    assert(std::to_integer<unsigned char>(y[2]) == 9);

    bool rejected = false;
    try {
        (void)reader.read_tensor("missing");
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);

    std::remove(path);
    std::cout << "safetensors tests passed\n";
}
