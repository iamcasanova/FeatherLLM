#include "featherllm/storage/mapped_file.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace featherllm::storage {
namespace {

[[noreturn]] void io_error(const std::string& message) {
    throw std::runtime_error("FeatherLLM mapped storage: " + message);
}

} // namespace

MappedFileRegion::MappedFileRegion(const std::byte* data,
                                   const std::size_t size,
                                   void* mapping_handle,
                                   const std::size_t mapped_size) noexcept
    : data_(data),
      size_(size),
      mapping_handle_(mapping_handle),
      mapped_size_(mapped_size) {}

MappedFileRegion::~MappedFileRegion() {
    if (mapping_handle_ == nullptr) return;
#ifdef _WIN32
    UnmapViewOfFile(mapping_handle_);
#else
    munmap(mapping_handle_, mapped_size_);
#endif
}

MappedFileReader::MappedFileReader(std::filesystem::path path)
    : path_(std::move(path)) {
    open();
}

MappedFileReader::~MappedFileReader() {
    close();
}

MappedFileReader::MappedFileReader(MappedFileReader&& other) noexcept
    : path_(std::move(other.path_)),
      size_(other.size_)
#ifdef _WIN32
      , handle_(other.handle_)
#else
      , fd_(other.fd_)
#endif
{
#ifdef _WIN32
    other.handle_ = nullptr;
#else
    other.fd_ = -1;
#endif
    other.size_ = 0;
}

MappedFileReader& MappedFileReader::operator=(MappedFileReader&& other) noexcept {
    if (this == &other) return *this;

    close();
    path_ = std::move(other.path_);
    size_ = other.size_;
#ifdef _WIN32
    handle_ = other.handle_;
    other.handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
    other.size_ = 0;
    return *this;
}

void MappedFileReader::open() {
#ifdef _WIN32
    HANDLE file = CreateFileW(
        path_.wstring().c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        io_error("failed to open " + path_.string());
    }

    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0) {
        CloseHandle(file);
        io_error("failed to stat " + path_.string());
    }

    handle_ = file;
    size_ = static_cast<std::uint64_t>(file_size.QuadPart);
#else
    fd_ = ::open(path_.c_str(), O_RDONLY);
    if (fd_ < 0) {
        io_error("failed to open " + path_.string());
    }

    struct stat file_stat {};
    if (fstat(fd_, &file_stat) != 0 || file_stat.st_size < 0) {
        close();
        io_error("failed to stat " + path_.string());
    }

    size_ = static_cast<std::uint64_t>(file_stat.st_size);
#endif
}

void MappedFileReader::close() noexcept {
#ifdef _WIN32
    if (handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
    }
#else
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
}

std::shared_ptr<const MappedFileRegion> MappedFileReader::map(
    const std::uint64_t offset,
    const std::size_t length) const {
    if (offset > size_ || length > size_ - offset) {
        throw std::out_of_range("mapped file range outside file");
    }
    if (length == 0) {
        return std::make_shared<const MappedFileRegion>(nullptr, 0, nullptr, 0);
    }

#ifdef _WIN32
    SYSTEM_INFO system_info{};
    GetSystemInfo(&system_info);
    const std::uint64_t granularity = system_info.dwAllocationGranularity;
    const std::uint64_t aligned_offset = offset - (offset % granularity);
    const std::uint64_t delta = offset - aligned_offset;
    if (delta > std::numeric_limits<std::size_t>::max() - length) {
        throw std::overflow_error("mapped file range is too large");
    }
    const std::size_t mapped_size = static_cast<std::size_t>(delta) + length;

    HANDLE mapping = CreateFileMappingW(
        static_cast<HANDLE>(handle_), nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        io_error("failed to create file mapping");
    }

    const auto high = static_cast<DWORD>(aligned_offset >> 32);
    const auto low = static_cast<DWORD>(aligned_offset & 0xffffffffu);
    void* view = MapViewOfFile(
        mapping, FILE_MAP_READ, high, low, mapped_size);
    if (view == nullptr) {
        CloseHandle(mapping);
        io_error("failed to map file range");
    }

    CloseHandle(mapping);
    return std::shared_ptr<const MappedFileRegion>(
        new MappedFileRegion(
            static_cast<const std::byte*>(view) + delta,
            length,
            view,
            mapped_size));
#else
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        io_error("failed to determine system page size");
    }
    const auto page = static_cast<std::uint64_t>(page_size);
    const std::uint64_t aligned_offset = offset - (offset % page);
    const std::uint64_t delta = offset - aligned_offset;
    if (aligned_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw std::out_of_range("mapped file offset exceeds platform off_t");
    }
    if (delta > std::numeric_limits<std::size_t>::max() - length) {
        throw std::overflow_error("mapped file range is too large");
    }
    const std::size_t mapped_size = static_cast<std::size_t>(delta) + length;

    void* mapping = ::mmap(
        nullptr,
        mapped_size,
        PROT_READ,
        MAP_PRIVATE,
        fd_,
        static_cast<off_t>(aligned_offset));
    if (mapping == MAP_FAILED) {
        io_error("failed to map file range");
    }

    return std::shared_ptr<const MappedFileRegion>(
        new MappedFileRegion(
            static_cast<const std::byte*>(mapping) + delta,
            length,
            mapping,
            mapped_size));
#endif
}

} // namespace featherllm::storage
