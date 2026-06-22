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

    let dst = cmake_cfg.build();

    // The build output directory.
    let build_dir = dst.join("build");

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
    // Point the linker at the directories containing libtts-cpp.a and the
    // GGML static archives.

    // tts-cpp library
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=tts-cpp");

    // mtl_tokenizer — compiled as a separate static library by cmake
    println!("cargo:rustc-link-lib=static=mtl_tokenizer");

    // GGML libraries — the cmake build put them in a flat build tree.
    let ggml_build_dir = build_dir.join("ggml/src");
    // Fallback for some cmake generators that use a deeper layout.
    let ggml_build_dir_alt = build_dir.join("_deps/ggml-build/src");

    for dir in &[&ggml_build_dir, &ggml_build_dir_alt] {
        if dir.join("libggml.a").exists() || dir.join("ggml.lib").exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    // Find libggml*.a in the build tree.
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");

    let cpu_lib = ggml_build_dir.join("ggml-cpu/libggml-cpu.a");
    let cpu_lib_alt = ggml_build_dir.join("libggml-cpu.a");
    if cpu_lib.exists() {
        println!("cargo:rustc-link-search=native={}", ggml_build_dir.join("ggml-cpu").display());
    } else if cpu_lib_alt.exists() {
        // Already in ggml_build_dir.
    }
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    if use_cuda {
        let cuda_dir = ggml_build_dir.join("ggml-cuda");
        let cuda_dir_alt = build_dir.join("ggml/src/ggml-cuda"); // different generator layout
        for dir in &[&cuda_dir, &cuda_dir_alt] {
            if dir.join("libggml-cuda.a").exists() || dir.join("ggml-cuda.lib").exists() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }
        println!("cargo:rustc-link-lib=static=ggml-cuda");

        // CUDA runtime libraries.
        let cuda_lib_dir = PathBuf::from("/usr/local/cuda/lib64");
        let cuda_stubs_dir = cuda_lib_dir.join("stubs");
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        println!("cargo:rustc-link-search=native={}", cuda_stubs_dir.display());
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // System libraries required by GGML and tts-cpp.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // OpenMP — ggml-cpu uses it.
    println!("cargo:rustc-link-lib=dylib=gomp");

    // Rerun if the C bridge or any chatterbox-cpp source changes.
    println!("cargo:rerun-if-changed=c_src/tts_bridge.cpp");
    println!("cargo:rerun-if-changed=c_src/tts_bridge.h");
    println!("cargo:rerun-if-changed={}", chatterbox_dir.join("src").display());
    println!("cargo:rerun-if-changed={}", chatterbox_dir.join("include").display());
    println!("cargo:rerun-if-changed={}", chatterbox_dir.join("ggml/include").display());
}
