#include <jni.h>

#include "featherllm/storage/checkpoint_manifest.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace {
constexpr const char* kRuntimeVersion = "FeatherLLM Android native runtime 0.1.0";

jstring make_string(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}

char hex_digit(unsigned value) {
    constexpr char digits[] = "0123456789abcdef";
    return digits[value & 0xfU];
}

std::string bytes_to_hex(const std::vector<std::byte>& bytes, std::size_t limit) {
    const auto count = std::min(bytes.size(), limit);
    std::string result;
    result.reserve(count * 2);
    for (std::size_t i = 0; i < count; ++i) {
        const auto value = std::to_integer<unsigned>(bytes[i]);
        result.push_back(hex_digit(value >> 4));
        result.push_back(hex_digit(value));
    }
    return result;
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_featherllm_android_MainActivity_nativeRuntimeVersion(JNIEnv* env, jclass) {
    return make_string(env, kRuntimeVersion);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_featherllm_android_MainActivity_nativeLoadCheckpointIndex(
    JNIEnv* env, jclass, jstring index_path) {
    if (index_path == nullptr) {
        return make_string(env, "error: checkpoint index path is null");
    }

    const char* path_chars = env->GetStringUTFChars(index_path, nullptr);
    if (path_chars == nullptr) {
        return make_string(env, "error: unable to read checkpoint index path");
    }

    const std::filesystem::path path(path_chars);
    env->ReleaseStringUTFChars(index_path, path_chars);

    try {
        const auto manifest = featherllm::storage::load_safetensors_index(path);
        return make_string(env, "loaded checkpoint index: " +
            std::to_string(manifest.tensors.size()) + " tensors");
    } catch (const std::exception& error) {
        return make_string(env, std::string("error: ") + error.what());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_featherllm_android_MainActivity_nativeReadTensorPrefix(
    JNIEnv* env, jclass, jstring index_path, jstring tensor_name, jint max_bytes) {
    if (index_path == nullptr || tensor_name == nullptr) {
        return make_string(env, "error: index path and tensor name are required");
    }
    if (max_bytes <= 0) {
        return make_string(env, "error: max_bytes must be positive");
    }

    const char* path_chars = env->GetStringUTFChars(index_path, nullptr);
    const char* tensor_chars = env->GetStringUTFChars(tensor_name, nullptr);
    if (path_chars == nullptr || tensor_chars == nullptr) {
        if (path_chars != nullptr) env->ReleaseStringUTFChars(index_path, path_chars);
        if (tensor_chars != nullptr) env->ReleaseStringUTFChars(tensor_name, tensor_chars);
        return make_string(env, "error: unable to read JNI strings");
    }

    const std::filesystem::path path(path_chars);
    const std::string tensor(tensor_chars);
    env->ReleaseStringUTFChars(index_path, path_chars);
    env->ReleaseStringUTFChars(tensor_name, tensor_chars);

    try {
        const auto bytes = featherllm::storage::read_tensor_bytes(
            path, tensor, 0, static_cast<std::size_t>(max_bytes));
        return make_string(env, "read " + std::to_string(bytes.size()) +
            " bytes; prefix_hex=" + bytes_to_hex(bytes, 64));
    } catch (const std::exception& error) {
        return make_string(env, std::string("error: ") + error.what());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_featherllm_android_MainActivity_nativeBenchmarkTensorRead(
    JNIEnv* env, jclass, jstring index_path, jstring tensor_name, jint max_bytes) {
    if (index_path == nullptr || tensor_name == nullptr) {
        return make_string(env, "error: index path and tensor name are required");
    }
    if (max_bytes <= 0) {
        return make_string(env, "error: max_bytes must be positive");
    }

    const char* path_chars = env->GetStringUTFChars(index_path, nullptr);
    const char* tensor_chars = env->GetStringUTFChars(tensor_name, nullptr);
    if (path_chars == nullptr || tensor_chars == nullptr) {
        if (path_chars != nullptr) env->ReleaseStringUTFChars(index_path, path_chars);
        if (tensor_chars != nullptr) env->ReleaseStringUTFChars(tensor_name, tensor_chars);
        return make_string(env, "error: unable to read JNI strings");
    }

    const std::filesystem::path path(path_chars);
    const std::string tensor(tensor_chars);
    env->ReleaseStringUTFChars(index_path, path_chars);
    env->ReleaseStringUTFChars(tensor_name, tensor_chars);

    try {
        const auto start = std::chrono::steady_clock::now();
        const auto bytes = featherllm::storage::read_tensor_bytes(
            path, tensor, 0, static_cast<std::size_t>(max_bytes));
        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        const double mib_per_second = elapsed_us > 0
            ? (static_cast<double>(bytes.size()) * 1000000.0) /
              (static_cast<double>(elapsed_us) * 1024.0 * 1024.0)
            : 0.0;

        return make_string(env,
            "bytes=" + std::to_string(bytes.size()) +
            ";elapsed_us=" + std::to_string(elapsed_us) +
            ";mib_per_s=" + std::to_string(mib_per_second));
    } catch (const std::exception& error) {
        return make_string(env, std::string("error: ") + error.what());
    }
}
