#include "featherllm/safetensors/tensor_reader.hpp"

#include <limits>
#include <stdexcept>

namespace featherllm::safetensors {

ShardedTensorReader::ShardedTensorReader(std::filesystem::path index_path,
                                         std::size_t window_bytes)
    : index_path_(std::move(index_path)),
      window_bytes_(window_bytes),
      manifest_(storage::load_safetensors_index(index_path_)) {
    if (window_bytes_ == 0) throw std::invalid_argument("window size must be non-zero");
}

std::vector<std::byte> ShardedTensorReader::read_tensor(const std::string& name) {
    const auto it = manifest_.tensors.find(name);
    if (it == manifest_.tensors.end()) throw std::out_of_range("tensor not found: " + name);

    const auto& location = it->second;
    if (location.data_length > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("tensor is too large for this process");

    auto reader_it = readers_.find(location.shard);
    if (reader_it == readers_.end()) {
        const auto path = index_path_.parent_path() / location.shard;
        auto [inserted, ok] = readers_.try_emplace(location.shard, path, window_bytes_);
        (void)ok;
        reader_it = inserted;
    }

    std::vector<std::byte> data(static_cast<std::size_t>(location.data_length));
    reader_it->second.read(location.data_offset, data);
    return data;
}

} // namespace featherllm::safetensors
