#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace featherllm::storage {

class MappedFileRegion {
public:
    ~MappedFileRegion();

    MappedFileRegion(const MappedFileRegion&) = delete;
    MappedFileRegion& operator=(const MappedFileRegion&) = delete;
    MappedFileRegion(MappedFileRegion&&) = delete;
    MappedFileRegion& operator=(MappedFileRegion&&) = delete;

    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

public:
    // Constructs an unmapped region. Only used for zero-length views.
    MappedFileRegion(const std::byte* data,
                     std::size_t size,
                     void* mapping_handle,
                     std::size_t mapped_size) noexcept;

private:
    friend class MappedFileReader;

    MappedFileRegion(const std::byte* data,
                     std::size_t size,
                     void* mapping_handle,
                     std::size_t mapped_size) noexcept;

    const std::byte* data_{nullptr};
    std::size_t size_{0};
    void* mapping_handle_{nullptr};
    std::size_t mapped_size_{0};
};

class MappedFileReader {
public:
    explicit MappedFileReader(std::filesystem::path path);
    ~MappedFileReader();

    MappedFileReader(const MappedFileReader&) = delete;
    MappedFileReader& operator=(const MappedFileReader&) = delete;
    MappedFileReader(MappedFileReader&&) noexcept;
    MappedFileReader& operator=(MappedFileReader&&) noexcept;

    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }

    // Maps exactly the requested byte range while internally aligning the
    // platform mapping offset. The returned region owns its mapping.
    [[nodiscard]] std::shared_ptr<const MappedFileRegion> map(
        std::uint64_t offset,
        std::size_t length) const;

private:
    void close() noexcept;
    void open();

    std::filesystem::path path_;
    std::uint64_t size_{0};
#ifdef _WIN32
    void* handle_{nullptr};
#else
    int fd_{-1};
#endif
};

} // namespace featherllm::storage
