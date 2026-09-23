cmake_minimum_required(VERSION 3.20)
project(FeatherLLM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(featherllm_storage
    src/storage/bounded_file.cpp
)
target_include_directories(featherllm_storage PUBLIC include)

add_library(featherllm_safetensors
    src/safetensors.cpp
    src/storage/checkpoint_manifest.cpp
    src/safetensors/tensor_reader.cpp
)
target_include_directories(featherllm_safetensors PUBLIC include)
target_link_libraries(featherllm_safetensors PUBLIC featherllm_storage)

enable_testing()

add_executable(bounded_file_tests tests/bounded_file_test.cpp)
target_link_libraries(bounded_file_tests PRIVATE featherllm_storage)
add_test(NAME bounded_file_tests COMMAND bounded_file_tests)

add_executable(safetensors_tests tests/safetensors_tests.cpp)
target_link_libraries(safetensors_tests PRIVATE featherllm_safetensors)
add_test(NAME safetensors_tests COMMAND safetensors_tests)

add_executable(checkpoint_manifest_tests tests/checkpoint_manifest_tests.cpp)
target_link_libraries(checkpoint_manifest_tests PRIVATE featherllm_safetensors)
add_test(NAME checkpoint_manifest_tests COMMAND checkpoint_manifest_tests)

add_executable(tensor_reader_tests tests/tensor_reader_tests.cpp)
target_link_libraries(tensor_reader_tests PRIVATE featherllm_safetensors)
add_test(NAME tensor_reader_tests COMMAND tensor_reader_tests)

add_executable(storage_benchmark benchmarks/storage_benchmark.cpp)
target_link_libraries(storage_benchmark PRIVATE featherllm_storage)
