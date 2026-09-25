#include "featherllm/storage/mapped_file.hpp"

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <fstream>

int main() {
    const auto path = std::filesystem::current_path() / "mapped_file_test.bin";
    {
        std::ofstream out(path, std::ios::binary);
        for (unsigned value = 0; value < 32; ++value) {
            const auto byte = static_cast<unsigned char>(value);
            out.write(reinterpret_cast<const char*>(&byte), 1);
        }
    }

    featherllm::storage::MappedFileReader reader(path);
    assert(reader.size() == 32);

    const auto region = reader.map(3, 7);
    assert(region);
    assert(region->size() == 7);
    for (std::size_t i = 0; i < region->size(); ++i) {
        assert(region->data()[i] == static_cast<std::byte>(i + 3));
    }

    const auto retained = reader.map(16, 4);
    assert(retained->data()[0] == std::byte{16});
    assert(retained->data()[3] == std::byte{19});

    const auto empty = reader.map(32, 0);
    assert(empty);
    assert(empty->empty());
    assert(empty->data() == nullptr);

    bool rejected = false;
    try {
        (void)reader.map(30, 3);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);

    std::filesystem::remove(path);
    return 0;
}
