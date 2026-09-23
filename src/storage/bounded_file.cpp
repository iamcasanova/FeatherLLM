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
namespace { [[noreturn]] void io_error(const std::string& m){ throw std::runtime_error("FeatherLLM storage: "+m); } }
BoundedFileReader::BoundedFileReader(std::filesystem::path path,std::size_t window_bytes):path_(std::move(path)),window_bytes_(window_bytes){
  if(!window_bytes_) throw std::invalid_argument("window_bytes must be non-zero"); open();
}
BoundedFileReader::~BoundedFileReader(){close();}
BoundedFileReader::BoundedFileReader(BoundedFileReader&& o) noexcept:path_(std::move(o.path_)),window_bytes_(o.window_bytes_),size_(o.size_)
#ifdef _WIN32
,handle_(o.handle_)
#else
,fd_(o.fd_)
#endif
{
#ifdef _WIN32
 o.handle_=nullptr;
#else
 o.fd_=-1;
#endif
 o.size_=0;
}
BoundedFileReader& BoundedFileReader::operator=(BoundedFileReader&& o) noexcept{
 if(this==&o)return *this; close(); path_=std::move(o.path_); window_bytes_=o.window_bytes_; size_=o.size_;
#ifdef _WIN32
 handle_=o.handle_; o.handle_=nullptr;
#else
 fd_=o.fd_; o.fd_=-1;
#endif
 o.size_=0; return *this;
}
void BoundedFileReader::open(){
#ifdef _WIN32
 HANDLE f=CreateFileW(path_.wstring().c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
 if(f==INVALID_HANDLE_VALUE) io_error("failed to open "+path_.string());
 LARGE_INTEGER s{};
 if(!GetFileSizeEx(f,&s)||s.QuadPart<0){CloseHandle(f);io_error("failed to stat "+path_.string());}
 handle_=f; size_=static_cast<std::uint64_t>(s.QuadPart);
#else
 fd_=::open(path_.c_str(),O_RDONLY); if(fd_<0)io_error("failed to open "+path_.string());
 struct stat s{}; if(fstat(fd_,&s)!=0||s.st_size<0){close();io_error("failed to stat "+path_.string());}
 size_=static_cast<std::uint64_t>(s.st_size);
#endif
}
void BoundedFileReader::close() noexcept{
#ifdef _WIN32
 if(handle_){CloseHandle(static_cast<HANDLE>(handle_));handle_=nullptr;}
#else
 if(fd_>=0){::close(fd_);fd_=-1;}
#endif
}
void BoundedFileReader::read(std::uint64_t offset,std::vector<std::byte>& dst){
 if(offset>size_||dst.size()>size_-offset) throw std::out_of_range("file range outside file");
 std::size_t done=0;
 while(done<dst.size()){
  const auto n=std::min(window_bytes_,dst.size()-done);
#ifdef _WIN32
  LARGE_INTEGER p{}; p.QuadPart=static_cast<LONGLONG>(offset+done);
  if(!SetFilePointerEx(static_cast<HANDLE>(handle_),p,nullptr,FILE_BEGIN))io_error("seek failed");
  DWORD got=0; if(!ReadFile(static_cast<HANDLE>(handle_),dst.data()+done,static_cast<DWORD>(n),&got,nullptr))io_error("read failed");
  if(!got)io_error("unexpected EOF"); done+=got;
#else
  const auto got=::pread(fd_,dst.data()+done,n,static_cast<off_t>(offset+done));
  if(got<0)io_error("read failed"); if(got==0)io_error("unexpected EOF"); done+=static_cast<std::size_t>(got);
#endif
 }
}
}
