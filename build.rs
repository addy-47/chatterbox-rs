use std::path::{Path, PathBuf};
use std::process::Command;

fn has_cuda() -> bool {
    if Path::new("/usr/local/cuda/bin/nvcc").exists() {
        return true;
    }
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            return true;
        }
    }
    false
}

/// Emit `cargo:rustc-link-search=native=<dir>` for every directory in `candidates`
/// that actually exists. This avoids spurious "no such directory" linker warnings
/// on generators / platforms that use only a subset of these layouts.
fn emit_link_search(candidates: &[PathBuf]) {
    for dir in candidates {
        if dir.exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
}

fn main() {
    // Path to the vendored chatterbox-cpp source.
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let chatterbox_dir = manifest_dir.join("chatterbox-cpp");

    // ── Step 1: Build libtts-cpp.a (and GGML static libs) via cmake ──
    //
    // We build GGML as static libraries (BUILD_SHARED_LIBS=OFF) so the Rust
    // binary doesn't need to ship or find .so files at runtime.
    let mut cmake_cfg = cmake::Config::new(&chatterbox_dir);
    cmake_cfg
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("TTS_CPP_BUILD_LIBRARY", "ON")
        .define("TTS_CPP_INSTALL", "OFF")
        .profile("Release");

    // CUDA is only compiled when the "cuda" feature is active AND nvcc is found.
    let cuda_feature = std::env::var("CARGO_FEATURE_CUDA").is_ok();
    let use_cuda = cuda_feature && has_cuda();
    if use_cuda {
        println!("cargo:warning=CUDA detected, building chatterbox-cpp with GPU support");
        cmake_cfg
            .define("GGML_CUDA", "ON")
            .define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc")
            .define("CMAKE_CUDA_ARCHITECTURES", "native");
    } else {
        println!("cargo:warning=CUDA not detected, building chatterbox-cpp CPU-only");
    }

    // Disable CUDA graphs (llama-only feature, not needed).
    cmake_cfg.define("GGML_CUDA_GRAPHS", "OFF");
    // Disable OpenMP to prevent linker errors and runtime thread contention with Vox's OpenMP runtime.
    cmake_cfg.define("GGML_OPENMP", "OFF");

    // ── Android (aarch64) toolchain ──
    //
    // Without a CMAKE_TOOLCHAIN_FILE, CMake resolves the cross-compile target on
    // its own and `ggml_get_system_arch()` (ggml/cmake/common.cmake) sees the
    // HOST processor instead of the target. It then selects its x86 backend:
    //
    //   -- CMAKE_SYSTEM_PROCESSOR: x86_64
    //   -- x86 detected
    //   -- Adding CPU backend variant ggml-cpu: -march=native
    //   clang: error: unsupported argument 'native' to option '-march='
    //
    // Passing `-DCMAKE_SYSTEM_PROCESSOR=aarch64` alone does NOT help: CMake
    // recomputes that variable inside `project()` and shadows the cache entry.
    // Only the NDK toolchain file sets it early enough to be authoritative.
    //
    // Mirrors llama-cpp-sys-4/build.rs:1885-1901 so both ggml builds agree.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("android") && target.contains("aarch64") {
        let android_ndk = std::env::var("ANDROID_NDK").expect(
            "Android target requires the NDK: set ANDROID_NDK (e.g. \
             ~/Android/Sdk/ndk/29.0.13846066)",
        );
        cmake_cfg
            .define(
                "CMAKE_TOOLCHAIN_FILE",
                format!("{android_ndk}/build/cmake/android.toolchain.cmake"),
            )
            .define("ANDROID_ABI", "arm64-v8a")
            // Defaults to android-24 (Vox's minSdk); override to match your app.
            .define(
                "ANDROID_PLATFORM",
                std::env::var("ANDROID_PLATFORM").unwrap_or_else(|_| "android-24".to_owned()),
            )
            .define("CMAKE_SYSTEM_PROCESSOR", "arm64")
            .define("CMAKE_C_FLAGS", "-march=armv8.7a")
            .define("CMAKE_CXX_FLAGS", "-march=armv8.7a");
        println!("cargo:rerun-if-env-changed=ANDROID_NDK");
    }

    let dst = cmake_cfg.build();

    // The build output directory (cmake crate's `dst` equals the OUT_DIR for this crate).
    let build_dir = dst.join("build");

    // On Windows, MSVC multi-config generators (Visual Studio) place compiled
    // static libraries in `<build>/Release/` instead of directly in `<build>/`.
    // We register both so the same build.rs works on Linux, macOS, and Windows.
    let build_dir_release = build_dir.join("Release");

    // ── Step 2: Compile the C bridge with the `cc` crate ──
    //
    // tts_bridge.cpp wraps the C++ Engine in extern "C" functions.  It
    // needs to see the public and private headers of chatterbox-cpp.
    let bridge_src = manifest_dir.join("c_src/tts_bridge.cpp");

    let mut bridge = cc::Build::new();
    bridge
        .cpp(true)
        .file(&bridge_src)
        .include(chatterbox_dir.join("include"))     // engine.h (public)
        .include(chatterbox_dir.join("src"))          // chatterbox_t3_internal.h (private)
        .include(chatterbox_dir.join("ggml/include")) // ggml.h etc.
        .flag_if_supported("-std=c++17");

    if use_cuda {
        bridge.define("GGML_USE_CUDA", None);
    }

    bridge.compile("tts_bridge");

    // ── Step 3: Link paths ──
    //
    // Point the linker at the directories containing libtts-cpp.a/.lib and the
    // GGML static archives. We register both flat (Unix Makefile) and
    // Release-subdirectory (Windows MSVC multi-config) layouts.

    // tts-cpp library — flat layout (Linux/macOS) and MSVC Release sub-dir (Windows).
    emit_link_search(&[build_dir.clone(), build_dir_release.clone()]);
    println!("cargo:rustc-link-lib=static=tts-cpp");

    // mtl_tokenizer — compiled as a separate static library by cmake.
    // Also resides in `Release/` on MSVC generators.
    println!("cargo:rustc-link-lib=static=mtl_tokenizer");

    // ── GGML libraries ──
    //
    // The cmake build places them in different sub-trees depending on the
    // generator and how ggml is configured:
    //
    // Unix Makefile (Linux/macOS):
    //   build/ggml/src/                  (flat)
    //   build/ggml/src/ggml-cpu/         (ggml-cpu sub-library)
    //
    // Visual Studio (Windows):
    //   build/ggml/src/Release/          (multi-config top)
    //   build/ggml/src/ggml-cpu/Release/ (ggml-cpu sub-library)
    //   build/Release/                   (some generators flatten everything here)
    //
    // FetchContent fallback (some cmake setups):
    //   build/_deps/ggml-build/src/

    let ggml_src = build_dir.join("ggml/src");
    let ggml_src_release = ggml_src.join("Release");
    let ggml_cpu_dir = ggml_src.join("ggml-cpu");
    let ggml_cpu_release = ggml_cpu_dir.join("Release");
    let ggml_deps_src = build_dir.join("_deps/ggml-build/src");
    let ggml_deps_src_release = ggml_deps_src.join("Release");

    emit_link_search(&[
        ggml_src.clone(),
        ggml_src_release.clone(),
        ggml_cpu_dir.clone(),
        ggml_cpu_release.clone(),
        ggml_deps_src.clone(),
        ggml_deps_src_release.clone(),
    ]);

    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    if use_cuda {
        let cuda_dir = ggml_src.join("ggml-cuda");
        let cuda_dir_release = cuda_dir.join("Release");
        emit_link_search(&[cuda_dir, cuda_dir_release]);
        println!("cargo:rustc-link-lib=static=ggml-cuda");

        // CUDA runtime libraries.
        let cuda_lib_dir = PathBuf::from("/usr/local/cuda/lib64");
        let cuda_stubs_dir = cuda_lib_dir.join("stubs");
        emit_link_search(&[cuda_lib_dir, cuda_stubs_dir]);
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // System libraries required by GGML and tts-cpp will be resolved by the top-level target/linker.

    // Rerun if the C bridge or any chatterbox-cpp source changes.
    println!("cargo:rerun-if-changed=c_src/tts_bridge.cpp");
    println!("cargo:rerun-if-changed=c_src/tts_bridge.h");
    println!("cargo:rerun-if-changed={}", chatterbox_dir.join("src").display());
    println!("cargo:rerun-if-changed={}", chatterbox_dir.join("include").display());
    println!("cargo:rerun-if-changed={}", chatterbox_dir.join("ggml/include").display());
}
