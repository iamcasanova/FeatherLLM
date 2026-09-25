#include "featherllm/safetensors.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

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
    const std::string needle = std::string(""") + field + """;
    const auto field_pos = find_required(body, needle, 0, "missing tensor string field");
    std::size_t cursor = field_pos + needle.size();
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor]))) {
        ++cursor;
    }
    if (cursor >= body.size() || body[cursor++] != ':')
        throw std::runtime_error("malformed tensor string field");
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor]))) {
        ++cursor;
    }
    if (cursor >= body.size() || body[cursor++] != '"')
        throw std::runtime_error("tensor string field is not a JSON string");

    std::string value;
    bool escaped = false;
    while (cursor < body.size()) {
        const char c = body[cursor++];
        if (escaped) {
            // dtype strings are identifiers; accepting only the JSON escapes
            // needed for a string while preserving the value avoids silently
            // interpreting malformed escape sequences as dtype names.
            if (c == '"' || c == '\\' || c == '/')
                value.push_back(c);
            else
                throw std::runtime_error("unsupported escape in tensor string field");
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') return value;
        if (static_cast<unsigned char>(c) < 0x20)
            throw std::runtime_error("unescaped control character in tensor string field");
        value.push_back(c);
    }
    throw std::runtime_error("unterminated tensor string field");
}

std::pair<std::uint64_t, std::uint64_t> parse_offsets(const std::string& body) {
    const std::string needle = ""data_offsets"";
    const auto field_pos = find_required(body, needle, 0, "missing data_offsets");
    std::size_t cursor = field_pos + needle.size();
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor]))) {
        ++cursor;
    }
    if (cursor >= body.size() || body[cursor++] != ':')
        throw std::runtime_error("malformed data_offsets");
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor]))) {
        ++cursor;
    }
    if (cursor >= body.size() || body[cursor++] != '[')
        throw std::runtime_error("data_offsets is not an array");

    auto parse_number = [&](std::uint64_t& value) {
        while (cursor < body.size() &&
               std::isspace(static_cast<unsigned char>(body[cursor]))) {
            ++cursor;
        }
        const auto begin = cursor;
        while (cursor < body.size() &&
               std::isdigit(static_cast<unsigned char>(body[cursor]))) {
            ++cursor;
        }
        if (begin == cursor) throw std::runtime_error("malformed data_offsets number");
        try {
            value = std::stoull(body.substr(begin, cursor - begin));
        } catch (const std::exception&) {
            throw std::runtime_error("data_offsets number overflows uint64");
        }
    };

    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    parse_number(begin);
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor]))) {
        ++cursor;
    }
    if (cursor >= body.size() || body[cursor++] != ',')
        throw std::runtime_error("data_offsets must contain two values");
    parse_number(end);
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor]))) {
        ++cursor;
    }
    if (cursor >= body.size() || body[cursor++] != ']')
        throw std::runtime_error("malformed data_offsets");
    return {begin, end};
}

std::vector<std::uint64_t> parse_shape(const std::string& body) {
    const std::string needle = ""shape"";
    const auto field_pos = find_required(body, needle, 0, "missing shape");
    std::size_t cursor = field_pos + needle.size();
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor]))) {
        ++cursor;
    }
    if (cursor >= body.size() || body[cursor++] != ':')
        throw std::runtime_error("malformed shape");
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor]))) {
        ++cursor;
    }
    if (cursor >= body.size() || body[cursor++] != '[')
        throw std::runtime_error("shape is not an array");

    std::vector<std::uint64_t> shape;
    while (true) {
        while (cursor < body.size() &&
               std::isspace(static_cast<unsigned char>(body[cursor]))) {
            ++cursor;
        }
        if (cursor >= body.size()) throw std::runtime_error("unterminated shape");
        if (body[cursor] == ']') {
            ++cursor;
            return shape;
        }

        const auto begin = cursor;
        while (cursor < body.size() &&
               std::isdigit(static_cast<unsigned char>(body[cursor]))) {
            ++cursor;
        }
        if (begin == cursor) throw std::runtime_error("malformed shape dimension");
        try {
            shape.push_back(std::stoull(body.substr(begin, cursor - begin)));
        } catch (const std::exception&) {
            throw std::runtime_error("shape dimension overflows uint64");
        }

        while (cursor < body.size() &&
               std::isspace(static_cast<unsigned char>(body[cursor]))) {
            ++cursor;
        }
        if (cursor >= body.size()) throw std::runtime_error("unterminated shape");
        if (body[cursor] == ',') {
            ++cursor;
            continue;
        }
        if (body[cursor] == ']') {
            ++cursor;
            return shape;
        }
        throw std::runtime_error("malformed shape separator");
    }
}

void validate_data_coverage(const std::unordered_map<std::string, TensorInfo>& tensors,
                            std::uint64_t data_bytes) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(tensors.size());
    for (const auto& [name, info] : tensors) {
        (void)name;
        ranges.emplace_back(info.data_begin, info.data_end);
    }
    std::sort(ranges.begin(), ranges.end());

    std::uint64_t cursor = 0;
    for (const auto [begin, end] : ranges) {
        if (begin != cursor)
            throw std::runtime_error("tensor ranges do not completely cover the data buffer");
        cursor = end;
    }
    if (cursor != data_bytes)
        throw std::runtime_error("tensor ranges do not completely cover the data buffer");
}

} // namespace

std::size_t dtype_size(const std::string& dtype) {
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" ||
        dtype == "F8_E4M3" || dtype == "F8_E4M3FNUZ" ||
        dtype == "F8_E5M2" || dtype == "F8_E5M2FNUZ" ||
        dtype == "F8_E8M0") return 1;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
    if (dtype == "F4_E2M1_X2") return 1;
    if (dtype == "C64") return 8;
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

    std::size_t first = 0;
    while (first < header.size() &&
           std::isspace(static_cast<unsigned char>(header[first]))) {
        ++first;
    }
    if (first >= header.size() || header[first] != '{')
        throw std::runtime_error("safetensors header must be a JSON object");

    std::size_t last = header.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(header[last - 1]))) {
        --last;
    }
    if (last <= first || header[last - 1] != '}')
        throw std::runtime_error("safetensors header must end with a JSON object");

    std::size_t pos = first + 1;
    while (pos < last - 1) {
        pos = header.find('"', pos);
        if (pos == std::string::npos || pos >= last - 1) break;
        const auto key_end = header.find('"', pos + 1);
        if (key_end == std::string::npos || key_end >= last)
            throw std::runtime_error("malformed JSON key");
        const std::string name = header.substr(pos + 1, key_end - pos - 1);
        const auto colon = header.find(':', key_end + 1);
        if (colon == std::string::npos || colon >= last)
            throw std::runtime_error("malformed JSON object member");
        const auto object_begin = header.find('{', colon + 1);
        if (object_begin == std::string::npos || object_begin >= last)
            throw std::runtime_error("malformed tensor object");
        const auto object_end = find_matching_object(header, object_begin);
        if (object_end >= last)
            throw std::runtime_error("malformed tensor object");

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

        if (!tensors_.emplace(name, std::move(info)).second)
            throw std::runtime_error("duplicate tensor name in safetensors header: " + name);
        pos = object_end + 1;
    }

    validate_data_coverage(tensors_, file_size_ - data_offset_);
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
