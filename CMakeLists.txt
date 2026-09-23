cmake_minimum_required(VERSION 3.20)
project(FeatherLLM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(featherllm_safetensors
    src/safetensors.cpp
)
target_include_directories(featherllm_safetensors PUBLIC include)

add_executable(safetensors_tests tests/safetensors_tests.cpp)
target_link_libraries(safetensors_tests PRIVATE featherllm_safetensors)

enable_testing()
add_test(NAME safetensors_tests COMMAND safetensors_tests)
