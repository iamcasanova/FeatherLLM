#include "featherllm/safetensors.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace featherllm::safetensors {
namespace {

std::uint64_t parse_u64_le(const unsigned char* p) {
    std::uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i) v |= std::uint64_t(p[i]) << (8 * i);
    return v;
}

std::uint64_t checked_product(const std::vector<std::uint64_t>& shape) {
    std::uint64_t n = 1;
    for (auto d : shape) {
        if (d != 0 && n > std::numeric_limits<std::uint64_t>::max() / d)
            throw std::runtime_error("tensor shape overflows uint64");
        n *= d;
    }
    return n;
}

} // namespace

std::size_t dtype_size(const std::string& dtype) {
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8") return 1;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
    throw std::runtime_error("unsupported dtype: " + dtype);
}

Reader::Reader(std::filesystem::path path) : path_(std::move(path)) {}

void Reader::open() {
    std::ifstream in(path_, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open safetensors file");
    const auto end = in.tellg();
    if (end < 8) throw std::runtime_error("safetensors file is smaller than header prefix");
    file_size_ = static_cast<std::uint64_t>(end);
    in.seekg(0);

    unsigned char prefix[8]{};
    in.read(reinterpret_cast<char*>(prefix), 8);
    if (!in) throw std::runtime_error("failed to read safetensors header length");
    const auto header_len = parse_u64_le(prefix);
    if (header_len > file_size_ - 8) throw std::runtime_error("invalid safetensors header length");

    std::string header(static_cast<std::size_t>(header_len), '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!in) throw std::runtime_error("failed to read safetensors header");

    data_offset_ = 8 + header_len;
    parse_header(header);
}

void Reader::parse_header(const std::string& header) {
    // Minimal strict parser for the tensor metadata subset. Full JSON support is
    // intentionally deferred to the storage-engine milestone rather than relying
    // on a third-party JSON dependency.
    tensors_.clear();

    std::size_t pos = 0;
    while (true) {
        pos = header.find('"', pos);
        if (pos == std::string::npos) break;
        const auto key_end = header.find('"', pos + 1);
        if (key_end == std::string::npos) throw std::runtime_error("malformed JSON key");
        const std::string name = header.substr(pos + 1, key_end - pos - 1);
        pos = key_end + 1;
        if (name == "__metadata__") continue;

        const auto obj = header.find('{', pos);
        if (obj == std::string::npos) throw std::runtime_error("malformed tensor object");
        const auto obj_end = header.find('}', obj);
        if (obj_end == std::string::npos) throw std::runtime_error("malformed tensor object");
        const std::string body = header.substr(obj, obj_end - obj + 1);

        auto find_string = [&](const char* field) -> std::string {
            const std::string needle = std::string(""") + field + "":"";
            const auto p = body.find(needle);
            if (p == std::string::npos) throw std::runtime_error("missing tensor field: " + std::string(field));
            const auto s = p + needle.size();
            const auto e = body.find('"', s);
            if (e == std::string::npos) throw std::runtime_error("malformed string field");
            return body.substr(s, e - s);
        };

        auto find_u64_pair = [&](const char* field) {
            const std::string needle = std::string(""") + field + "":[";
            const auto p = body.find(needle);
            if (p == std::string::npos) throw std::runtime_error("missing offsets");
            const auto s = p + needle.size();
            const auto comma = body.find(',', s);
            const auto e = body.find(']', comma);
            if (comma == std::string::npos || e == std::string::npos) throw std::runtime_error("malformed offsets");
            return std::pair<std::uint64_t,std::uint64_t>{
                std::stoull(body.substr(s, comma-s)),
                std::stoull(body.substr(comma+1, e-comma-1))
            };
        };

        TensorInfo info;
        info.dtype = find_string("dtype");
        const auto shape_pos = body.find(""shape":[");
        if (shape_pos == std::string::npos) throw std::runtime_error("missing shape");
        const auto s = shape_pos + 9;
        const auto e = body.find(']', s);
        if (e == std::string::npos) throw std::runtime_error("malformed shape");
        std::size_t cur = s;
        while (cur < e) {
            const auto comma = body.find(',', cur);
            const auto stop = comma == std::string::npos || comma > e ? e : comma;
            if (stop > cur) info.shape.push_back(std::stoull(body.substr(cur, stop-cur)));
            cur = stop + 1;
        }
        const auto [begin, end] = find_u64_pair("data_offsets");
        info.data_begin = begin;
        info.data_end = end;
        if (info.data_end < info.data_begin) throw std::runtime_error("invalid tensor range");

        const auto elements = checked_product(info.shape);
        const auto expected = elements * dtype_size(info.dtype);
        if (expected != info.data_end - info.data_begin)
            throw std::runtime_error("tensor byte range does not match dtype and shape");
        tensors_.emplace(name, std::move(info));
        pos = obj_end + 1;
    }
}

std::vector<std::byte> Reader::read_tensor(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::out_of_range("tensor not found: " + name);
    const auto& t = it->second;
    const auto bytes = t.data_end - t.data_begin;

    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open safetensors file");
    in.seekg(static_cast<std::streamoff>(data_offset_ + t.data_begin));
    if (!in) throw std::runtime_error("failed to seek tensor data");

    std::vector<std::byte> out(static_cast<std::size_t>(bytes));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("failed to read tensor data");
    return out;
}

} // namespace featherllm::safetensors
