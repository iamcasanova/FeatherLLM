#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace featherllm::safetensors {

struct TensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t data_begin{};
    std::uint64_t data_end{};
};

class Reader {
public:
    explicit Reader(std::filesystem::path path);

    void open();
    const std::unordered_map<std::string, TensorInfo>& tensors() const noexcept { return tensors_; }
    std::uint64_t data_offset() const noexcept { return data_offset_; }
    std::vector<std::byte> read_tensor(const std::string& name) const;

private:
    std::filesystem::path path_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    std::uint64_t data_offset_{};
    std::uint64_t file_size_{};

    void parse_header(const std::string& header);
};

std::size_t dtype_size(const std::string& dtype);

} // namespace featherllm::safetensors
