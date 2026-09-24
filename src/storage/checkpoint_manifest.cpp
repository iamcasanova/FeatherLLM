#include "featherllm/storage/checkpoint_manifest.hpp"

#include "featherllm/safetensors.hpp"
#include "featherllm/storage/bounded_file.hpp"

#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace featherllm::storage {
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open checkpoint index: " + path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void skip_ws(const std::string& text, std::size_t& cursor) {
    while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor]))) ++cursor;
}

unsigned parse_hex4(const std::string& text, std::size_t& cursor) {
    if (cursor + 4 > text.size()) throw std::runtime_error("truncated JSON unicode escape");
    unsigned value = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned char c = static_cast<unsigned char>(text[cursor++]);
        unsigned digit = 0;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else throw std::runtime_error("invalid JSON unicode escape");
        value = (value << 4) | digit;
    }
    return value;
}

void append_utf8(std::string& value, unsigned codepoint) {
    if (codepoint <= 0x7f) {
        value.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        value.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        value.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        value.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        value.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        value.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0x10ffff) {
        value.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        value.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        value.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        value.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        throw std::runtime_error("invalid JSON unicode code point");
    }
}

std::string parse_quoted(const std::string& text, std::size_t& cursor) {
    skip_ws(text, cursor);
    if (cursor >= text.size() || text[cursor] != '"')
        throw std::runtime_error("expected JSON string in weight_map");
    ++cursor;
    std::string value;
    while (cursor < text.size()) {
        const char c = text[cursor++];
        if (c == '"') return value;
        if (static_cast<unsigned char>(c) < 0x20)
            throw std::runtime_error("unescaped control character in JSON string");
        if (c != '\\') {
            value.push_back(c);
            continue;
        }
        if (cursor >= text.size()) throw std::runtime_error("unterminated JSON escape");
        switch (text[cursor++]) {
            case '"': value.push_back('"'); break;
            case '\\': value.push_back('\\'); break;
            case '/': value.push_back('/'); break;
            case 'b': value.push_back('\b'); break;
            case 'f': value.push_back('\f'); break;
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            case 'u': {
                const unsigned first = parse_hex4(text, cursor);
                unsigned codepoint = first;
                if (first >= 0xd800 && first <= 0xdbff) {
                    if (cursor + 6 > text.size() || text[cursor] != '\\' || text[cursor + 1] != 'u')
                        throw std::runtime_error("high surrogate without low surrogate");
                    cursor += 2;
                    const unsigned second = parse_hex4(text, cursor);
                    if (second < 0xdc00 || second > 0xdfff)
                        throw std::runtime_error("invalid low surrogate in JSON unicode escape");
                    codepoint = 0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00);
                } else if (first >= 0xdc00 && first <= 0xdfff) {
                    throw std::runtime_error("unexpected low surrogate in JSON unicode escape");
                }
                append_utf8(value, codepoint);
                break;
            }
            default: throw std::runtime_error("unsupported JSON escape in weight_map");
        }
    }
    throw std::runtime_error("unterminated JSON string in weight_map");
}

std::unordered_map<std::string, std::string> parse_weight_map(const std::string& json) {
    const std::string key = "\"weight_map\"";
    const auto key_pos = json.find(key);
    if (key_pos == std::string::npos) throw std::runtime_error("checkpoint index has no weight_map");

    std::size_t cursor = key_pos + key.size();
    skip_ws(json, cursor);
    if (cursor >= json.size() || json[cursor++] != ':')
        throw std::runtime_error("weight_map is missing its colon");
    skip_ws(json, cursor);
    if (cursor >= json.size() || json[cursor++] != '{')
        throw std::runtime_error("weight_map is not an object");

    std::unordered_map<std::string, std::string> result;
    skip_ws(json, cursor);
    if (cursor < json.size() && json[cursor] == '}') return result;

    while (true) {
        const auto tensor = parse_quoted(json, cursor);
        skip_ws(json, cursor);
        if (cursor >= json.size() || json[cursor++] != ':')
            throw std::runtime_error("malformed weight_map entry");
        const auto shard = parse_quoted(json, cursor);
        if (!result.emplace(tensor, shard).second)
            throw std::runtime_error("duplicate tensor in weight_map: " + tensor);

        skip_ws(json, cursor);
        if (cursor >= json.size()) throw std::runtime_error("unterminated weight_map");
        if (json[cursor] == '}') break;
        if (json[cursor++] != ',') throw std::runtime_error("malformed weight_map separator");
        skip_ws(json, cursor);
        if (cursor < json.size() && json[cursor] == '}')
            throw std::runtime_error("trailing comma in weight_map");
    }
    return result;
}

void validate_shard_path(const std::filesystem::path& shard) {
    if (shard.empty() || shard.is_absolute())
        throw std::runtime_error("checkpoint shard path must be relative");
    for (const auto& component : shard) {
        if (component == "..")
            throw std::runtime_error("checkpoint shard path escapes the index directory");
    }
}

} // namespace

CheckpointManifest load_safetensors_index(const std::filesystem::path& index_path) {
    const auto weight_map = parse_weight_map(read_text(index_path));
    const auto base = index_path.parent_path();

    std::unordered_map<std::string, safetensors::Reader> readers;
    CheckpointManifest manifest;
    for (const auto& [tensor, shard] : weight_map) {
        const std::filesystem::path shard_path(shard);
        validate_shard_path(shard_path);
        auto it = readers.find(shard);
        if (it == readers.end()) {
            safetensors::Reader reader(base / shard_path);
            reader.open();
            it = readers.emplace(shard, std::move(reader)).first;
        }
        const auto tensor_it = it->second.tensors().find(tensor);
        if (tensor_it == it->second.tensors().end())
            throw std::runtime_error("tensor " + tensor + " missing from shard " + shard);

        const auto& info = tensor_it->second;
        manifest.tensors.emplace(tensor, ShardTensorLocation{
            shard, it->second.data_offset() + info.data_begin, info.data_end - info.data_begin});
    }
    return manifest;
}

std::vector<std::byte> read_tensor_bytes(const std::filesystem::path& index_path,
                                         const std::string& tensor_name,
                                         std::uint64_t relative_offset,
                                         std::size_t max_bytes) {
    const auto manifest = load_safetensors_index(index_path);
    const auto it = manifest.tensors.find(tensor_name);
    if (it == manifest.tensors.end())
        throw std::runtime_error("tensor not found in checkpoint index: " + tensor_name);

    const auto& location = it->second;
    if (relative_offset > location.data_length)
        throw std::runtime_error("tensor read offset exceeds tensor range: " + tensor_name);
    const auto remaining = location.data_length - relative_offset;
    const auto requested = std::min<std::uint64_t>(remaining, max_bytes);
    if (requested > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("tensor read exceeds host size_t capacity");

    BoundedFileReader reader(index_path.parent_path() / location.shard);
    std::vector<std::byte> bytes(static_cast<std::size_t>(requested));
    if (!bytes.empty()) {
        reader.read(location.data_offset + relative_offset, bytes);
    }
    return bytes;
}

} // namespace featherllm::storage
