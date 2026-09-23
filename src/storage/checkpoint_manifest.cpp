#include "featherllm/storage/checkpoint_manifest.hpp"

#include "featherllm/safetensors.hpp"

#include <fstream>
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

std::string parse_quoted(const std::string& text, std::size_t& cursor) {
    while (cursor < text.size() && (text[cursor] == ' ' || text[cursor] == '\n' ||
                                    text[cursor] == '\r' || text[cursor] == '\t' ||
                                    text[cursor] == ',')) ++cursor;
    if (cursor >= text.size() || text[cursor] != '"')
        throw std::runtime_error("expected JSON string in weight_map");
    ++cursor;
    std::string value;
    bool escaped = false;
    while (cursor < text.size()) {
        const char c = text[cursor++];
        if (escaped) {
            value.push_back(c);
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            return value;
        } else {
            value.push_back(c);
        }
    }
    throw std::runtime_error("unterminated JSON string in weight_map");
}

std::unordered_map<std::string, std::string> parse_weight_map(const std::string& json) {
    const std::string key = "\"weight_map\"";
    const auto key_pos = json.find(key);
    if (key_pos == std::string::npos) throw std::runtime_error("checkpoint index has no weight_map");
    const auto object_begin = json.find('{', key_pos + key.size());
    if (object_begin == std::string::npos) throw std::runtime_error("weight_map is not an object");

    std::unordered_map<std::string, std::string> result;
    std::size_t cursor = object_begin + 1;
    while (true) {
        while (cursor < json.size() && (json[cursor] == ' ' || json[cursor] == '\n' ||
                                        json[cursor] == '\r' || json[cursor] == '\t' ||
                                        json[cursor] == ',')) ++cursor;
        if (cursor >= json.size()) throw std::runtime_error("unterminated weight_map");
        if (json[cursor] == '}') break;
        const auto tensor = parse_quoted(json, cursor);
        while (cursor < json.size() && (json[cursor] == ' ' || json[cursor] == '\n' ||
                                        json[cursor] == '\r' || json[cursor] == '\t')) ++cursor;
        if (cursor >= json.size() || json[cursor] != ':') throw std::runtime_error("malformed weight_map entry");
        ++cursor;
        const auto shard = parse_quoted(json, cursor);
        result.emplace(tensor, shard);
    }
    return result;
}

} // namespace

CheckpointManifest load_safetensors_index(const std::filesystem::path& index_path) {
    const auto weight_map = parse_weight_map(read_text(index_path));
    const auto base = index_path.parent_path();

    std::unordered_map<std::string, safetensors::Reader> readers;
    CheckpointManifest manifest;
    for (const auto& [tensor, shard] : weight_map) {
        auto it = readers.find(shard);
        if (it == readers.end()) {
            safetensors::Reader reader(base / shard);
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

} // namespace featherllm::storage
