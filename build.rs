use std::path::Path;
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
    let mut config = cmake::Config::new("codec.cpp");
    config.define("BUILD_SHARED_LIBS", "OFF");
    config.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
    config.profile("Release");

    let use_cuda = has_cuda();

    if use_cuda {
        println!("cargo:warning=CUDA detected, compiling codec.cpp with GPU support");
        config.define("GGML_CUDA", "ON");
        config.define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc");
        config.define("CMAKE_CUDA_ARCHITECTURES", "native");
    } else {
        println!("cargo:warning=CUDA not detected, compiling codec.cpp CPU-only");
        config.define("GGML_CUDA", "OFF");
    }

    let dst = config.build();

    // Link library search directories
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/build/ggml/src", dst.display());

    // Link libraries
    println!("cargo:rustc-link-lib=static=codec");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    if use_cuda {
        println!("cargo:rustc-link-search=native={}/build/ggml/src/ggml-cuda", dst.display());
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        
        // Link CUDA runtime and driver libraries
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // Link OpenMP for ggml-cpu
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
