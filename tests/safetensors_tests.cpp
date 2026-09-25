#include "featherllm/safetensors.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void write_file(const char* path, const std::string& header,
                const unsigned char* data, std::size_t data_size) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xff));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(data_size));
}

void expect_rejected(const char* path, const std::string& header,
                     const unsigned char* data, std::size_t data_size) {
    write_file(path, header, data, data_size);
    bool rejected = false;
    try {
        featherllm::safetensors::Reader reader(path);
        reader.open();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main() {
    const char* path = "featherllm_test.safetensors";
    const std::string header =
        "{"__metadata__":{"format":"pt","note":"test"},"
        ""x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"
        ""y":{"dtype":"U8","shape":[3],"data_offsets":[8,11]}}";
    const unsigned char payload[11] = {
        0, 0, 128, 63, 0, 0, 0, 64, 7, 8, 9};

    write_file(path, header, payload, sizeof(payload));

    featherllm::safetensors::Reader reader(path);
    reader.open();
    assert(reader.tensors().size() == 2);
    assert(reader.tensors().at("x").dtype == "F32");
    assert(reader.tensors().at("x").shape.size() == 1);
    assert(reader.tensors().at("x").shape[0] == 2);
    assert(reader.tensors().at("x").data_begin == 0);
    assert(reader.tensors().at("x").data_end == 8);

    const auto x = reader.read_tensor("x");
    assert(x.size() == sizeof(float) * 2);
    const auto y = reader.read_tensor("y");
    assert(y.size() == 3);
    assert(std::to_integer<unsigned char>(y[0]) == 7);
    assert(std::to_integer<unsigned char>(y[2]) == 9);

    bool rejected = false;
    try {
        (void)reader.read_tensor("missing");
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);

    const std::string whitespace_header =
        "{ "x" : { "dtype" : "U8", "shape" : [ 3 ], "
        ""data_offsets" : [ 0, 3 ] } }   ";
    write_file(path, whitespace_header, payload, 3);
    featherllm::safetensors::Reader whitespace_reader(path);
    whitespace_reader.open();
    assert(whitespace_reader.tensors().at("x").data_end == 3);

    expect_rejected(path,
        "{"x":{"dtype":"U8","shape":[3],"data_offsets":[0,2]}}",
        payload, 3);
    expect_rejected(path,
        "{"x":{"dtype":"U8","shape":[3],"data_offsets":[1,3]}}",
        payload, 3);
    expect_rejected(path,
        "{"x":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"
        ""y":{"dtype":"U8","shape":[0],"data_offsets":[3,3]},"
        ""x":{"dtype":"U8","shape":[0],"data_offsets":[3,3]}}",
        payload, 3);

    std::remove(path);
    std::cout << "safetensors tests passed
";
}
