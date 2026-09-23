#include "featherllm/storage/bounded_file.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <vector>
int main(){
 auto p=std::filesystem::temp_directory_path()/"featherllm_bounded_file_test.bin";
 {std::ofstream o(p,std::ios::binary|std::ios::trunc);for(int i=0;i<256;++i){auto v=static_cast<unsigned char>(i);o.write(reinterpret_cast<const char*>(&v),1);}}
 {featherllm::storage::BoundedFileReader r(p,7);assert(r.size()==256);std::vector<std::byte>d(19);r.read(101,d);for(std::size_t i=0;i<d.size();++i)assert(std::to_integer<unsigned char>(d[i])==static_cast<unsigned char>(101+i));}
 std::filesystem::remove(p); return 0;
}
