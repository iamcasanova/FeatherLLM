#include "featherllm/safetensors.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace featherllm::safetensors {
namespace {

std::uint64_t parse_u64_le(const unsigned char* p) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(p[i]) << (8u * i);
    return value;
}

std::uint64_t checked_product(const std::vector<std::uint64_t>& shape) {
    std::uint64_t result = 1;
    for (const auto dim : shape) {
        if (dim != 0 && result > std::numeric_limits<std::uint64_t>::max() / dim)
            throw std::runtime_error("tensor shape overflows uint64");
        result *= dim;
    }
    return result;
}

std::size_t find_required(const std::string& text, const std::string& needle,
                          std::size_t from, const char* error) {
    const auto pos = text.find(needle, from);
    if (pos == std::string::npos) throw std::runtime_error(error);
    return pos;
}

std::size_t find_matching_object(const std::string& text, std::size_t begin) {
    std::size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = begin; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') { in_string = true; continue; }
        if (c == '{') ++depth;
        else if (c == '}' && --depth == 0) return i;
    }
    throw std::runtime_error("unterminated JSON object");
}

std::string parse_json_string_field(const std::string& body, const char* field) {
    const std::string needle = std::string("\"") + field + "\":\"";
    const auto begin = find_required(body, needle, 0, "missing tensor string field") + needle.size();
    const auto end = body.find('"', begin);
    if (end == std::string::npos) throw std::runtime_error("unterminated tensor string field");
    return body.substr(begin, end - begin);
}

std::pair<std::uint64_t, std::uint64_t> parse_offsets(const std::string& body) {
    const std::string needle = "\"data_offsets\":[";
    const auto begin = find_required(body, needle, 0, "missing data_offsets") + needle.size();
    const auto comma = body.find(',', begin);
    const auto end = body.find(']', comma == std::string::npos ? begin : comma + 1);
    if (comma == std::string::npos || end == std::string::npos) throw std::runtime_error("malformed data_offsets");
    return {std::stoull(body.substr(begin, comma - begin)),
            std::stoull(body.substr(comma + 1, end - comma - 1))};
}

std::vector<std::uint64_t> parse_shape(const std::string& body) {
    const std::string needle = "\"shape\":[";
    const auto begin = find_required(body, needle, 0, "missing shape") + needle.size();
    const auto end = body.find(']', begin);
    if (end == std::string::npos) throw std::runtime_error("malformed shape");

    std::vector<std::uint64_t> shape;
    std::size_t cursor = begin;
    while (cursor < end) {
        while (cursor < end && (body[cursor] == ' ' || body[cursor] == ',')) ++cursor;
        if (cursor >= end) break;
        const auto comma = body.find(',', cursor);
        const auto stop = comma == std::string::npos || comma > end ? end : comma;
        shape.push_back(std::stoull(body.substr(cursor, stop - cursor)));
        cursor = stop + 1;
    }
    return shape;
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
    if (end < std::streamoff(8)) throw std::runtime_error("safetensors file is smaller than header prefix");
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
    tensors_.clear();
    std::size_t pos = 0;
    while (true) {
        pos = header.find('"', pos);
        if (pos == std::string::npos) break;
        const auto key_end = header.find('"', pos + 1);
        if (key_end == std::string::npos) throw std::runtime_error("malformed JSON key");
        const std::string name = header.substr(pos + 1, key_end - pos - 1);
        const auto colon = header.find(':', key_end + 1);
        if (colon == std::string::npos) throw std::runtime_error("malformed JSON object member");
        const auto object_begin = header.find('{', colon + 1);
        if (object_begin == std::string::npos) throw std::runtime_error("malformed tensor object");
        const auto object_end = find_matching_object(header, object_begin);

        if (name == "__metadata__") {
            pos = object_end + 1;
            continue;
        }

        const std::string body = header.substr(object_begin, object_end - object_begin + 1);
        TensorInfo info;
        info.dtype = parse_json_string_field(body, "dtype");
        info.shape = parse_shape(body);
        const auto [begin, end] = parse_offsets(body);
        info.data_begin = begin;
        info.data_end = end;
        if (info.data_end < info.data_begin) throw std::runtime_error("invalid tensor range");
        if (info.data_end > file_size_ - data_offset_) throw std::runtime_error("tensor range exceeds file data");

        const auto elements = checked_product(info.shape);
        const auto element_size = dtype_size(info.dtype);
        if (elements > std::numeric_limits<std::uint64_t>::max() / element_size)
            throw std::runtime_error("tensor byte size overflows uint64");
        if (elements * element_size != info.data_end - info.data_begin)
            throw std::runtime_error("tensor byte range does not match dtype and shape");

        tensors_.emplace(name, std::move(info));
        pos = object_end + 1;
    }
}

std::vector<std::byte> Reader::read_tensor(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::out_of_range("tensor not found: " + name);
    const auto& tensor = it->second;
    const auto bytes = tensor.data_end - tensor.data_begin;
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("tensor is too large for this process");

    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open safetensors file");
    in.seekg(static_cast<std::streamoff>(data_offset_ + tensor.data_begin));
    if (!in) throw std::runtime_error("failed to seek tensor data");
    std::vector<std::byte> out(static_cast<std::size_t>(bytes));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("failed to read tensor data");
    return out;
}

} // namespace featherllm::safetensors
