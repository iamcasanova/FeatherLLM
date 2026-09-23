#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>
namespace featherllm::storage {
class BoundedFileReader {
public:
  explicit BoundedFileReader(std::filesystem::path path, std::size_t window_bytes = 4 * 1024 * 1024);
  ~BoundedFileReader();
  BoundedFileReader(const BoundedFileReader&) = delete;
  BoundedFileReader& operator=(const BoundedFileReader&) = delete;
  BoundedFileReader(BoundedFileReader&&) noexcept;
  BoundedFileReader& operator=(BoundedFileReader&&) noexcept;
  [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t window_bytes() const noexcept { return window_bytes_; }
  void read(std::uint64_t offset, std::vector<std::byte>& dst);
private:
  void close() noexcept;
  void open();
  std::filesystem::path path_;
  std::size_t window_bytes_;
  std::uint64_t size_ = 0;
#ifdef _WIN32
  void* handle_ = nullptr;
#else
  int fd_ = -1;
#endif
};
}
