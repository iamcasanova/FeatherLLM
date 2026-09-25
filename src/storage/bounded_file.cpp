#include "featherllm/storage/bounded_file.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace featherllm::storage {
namespace {

[[noreturn]] void io_error(const std::string& message) {
    throw std::runtime_error("FeatherLLM storage: " + message);
}

} // namespace

BoundedFileReader::BoundedFileReader(std::filesystem::path path, std::size_t window_bytes)
    : path_(std::move(path)), window_bytes_(window_bytes) {
    if (window_bytes_ == 0) {
        throw std::invalid_argument("window_bytes must be non-zero");
    }
    open();
}

BoundedFileReader::~BoundedFileReader() {
    close();
}

BoundedFileReader::BoundedFileReader(BoundedFileReader&& other) noexcept
    : path_(std::move(other.path_)),
      window_bytes_(other.window_bytes_),
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

BoundedFileReader& BoundedFileReader::operator=(BoundedFileReader&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    close();
    path_ = std::move(other.path_);
    window_bytes_ = other.window_bytes_;
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

void BoundedFileReader::open() {
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

void BoundedFileReader::close() noexcept {
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

void BoundedFileReader::read(std::uint64_t offset, std::vector<std::byte>& dst) {
    if (offset > size_ || dst.size() > size_ - offset) {
        throw std::out_of_range("file range outside file");
    }

    std::size_t done = 0;
    while (done < dst.size()) {
        const auto chunk = std::min(window_bytes_, dst.size() - done);
#ifdef _WIN32
        LARGE_INTEGER position{};
        position.QuadPart = static_cast<LONGLONG>(offset + done);
        if (!SetFilePointerEx(
                static_cast<HANDLE>(handle_), position, nullptr, FILE_BEGIN)) {
            io_error("seek failed");
        }

        DWORD bytes_read = 0;
        if (!ReadFile(
                static_cast<HANDLE>(handle_),
                dst.data() + done,
                static_cast<DWORD>(chunk),
                &bytes_read,
                nullptr)) {
            io_error("read failed");
        }
        if (bytes_read == 0) {
            io_error("unexpected EOF");
        }
        done += bytes_read;
#else
        const auto bytes_read =
            ::pread(fd_, dst.data() + done, chunk, static_cast<off_t>(offset + done));
        if (bytes_read < 0) {
            io_error("read failed");
        }
        if (bytes_read == 0) {
            io_error("unexpected EOF");
        }
        done += static_cast<std::size_t>(bytes_read);
#endif
    }
}

} // namespace featherllm::storage
