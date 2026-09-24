cmake_minimum_required(VERSION 3.20)
project(FeatherLLMAndroid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Reuse the canonical FeatherLLM C++ build instead of duplicating source lists.
set(FEATHERLLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../../.. ${CMAKE_CURRENT_BINARY_DIR}/featherllm-core)

add_library(featherllm_android SHARED
    featherllm_jni.cpp
)

target_include_directories(featherllm_android PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../../../include
)

target_link_libraries(featherllm_android
    PRIVATE
    featherllm_safetensors
    android
    log
)
