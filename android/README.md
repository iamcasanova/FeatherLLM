# FeatherLLM Android

This directory is the first-class Android build path for the canonical FeatherLLM C++ core. It does not replace the desktop CMake build.

## Toolchain

- Android Gradle Plugin: 9.4.0
- Gradle: 9.6.0
- JDK: 17
- Android API: compile/target 36
- Android NDK LTS: r30 (`30.0.16248370`)
- Android CMake: 3.10.2 (SDK-managed)
- ABI: `arm64-v8a`

The NDK and CMake binaries are host/CI prerequisites and are intentionally not committed to this repository.

## Local setup

Install Android SDK command-line tools, then accept licenses and install the pinned packages:

```powershell
sdkmanager --licenses
sdkmanager "platform-tools" "platforms;android-36" "build-tools;36.0.0" "cmake;3.10.2" "ndk;30.0.16248370"
```

Ensure JDK 17 is active. From the repository root:

```powershell
cd android
gradle --version
gradle --no-daemon assembleDebug
```

The APK is produced under `android/app/build/outputs/apk/debug/`.

## Device install

With an ARM64 Android device connected and USB debugging enabled:

```powershell
adb devices
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The current milestone validates the complete Gradle -> CMake -> NDK -> ARM64 native library -> JNI -> APK path and exposes the native runtime version in the app. Full model loading/token generation is a subsequent milestone; this scaffold must not be represented as a completed inference runtime.

## Native build topology

`android/app/src/main/cpp/CMakeLists.txt` reuses the repository's top-level CMake project with `FEATHERLLM_BUILD_TESTS=OFF`, so Android does not maintain a second copy of FeatherLLM source lists. The desktop test/benchmark targets remain enabled by default.
