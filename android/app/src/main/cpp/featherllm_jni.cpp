#include <jni.h>

#include "featherllm/storage/checkpoint_manifest.hpp"

#include <filesystem>
#include <string>

namespace {
constexpr const char* kRuntimeVersion = "FeatherLLM Android native runtime 0.1.0";

jstring make_string(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_featherllm_android_MainActivity_nativeRuntimeVersion(JNIEnv* env, jobject) {
    return make_string(env, kRuntimeVersion);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_featherllm_android_MainActivity_nativeLoadCheckpointIndex(
    JNIEnv* env, jobject, jstring index_path) {
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
