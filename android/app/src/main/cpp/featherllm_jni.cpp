#include <jni.h>
#include <string>

namespace {
constexpr const char* kRuntimeVersion = "FeatherLLM Android native runtime 0.1.0";
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_featherllm_android_MainActivity_nativeRuntimeVersion(JNIEnv* env, jobject) {
    return env->NewStringUTF(kRuntimeVersion);
}
