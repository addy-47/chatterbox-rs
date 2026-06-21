# chatterbox-rs

Rust bindings for [chatterbox.cpp](https://github.com/addy-47/chatterbox.cpp) — a Multilingual TTS engine.

## Features

- **Two TTS variants**: Turbo (fast) and Multilingual (18+ languages)
- **GPU acceleration**: CUDA via GGML (CPU fallback)
- **Built-in reference voice** — no voice cloning needed
- **Optional HTTP/WS server** (axum) behind the `server` feature
- **Same C++ engine, zero-copy FFI** via extern "C" bridge

## Supported Languages

`en`, `es`, `fr`, `de`, `it`, `pt`, `nl`, `pl`, `tr`, `sv`, `da`, `fi`, `no`, `el`, `ms`, `sw`, `ar`, `ko`

## Quick Start

```bash
cargo build --example synthesize
cargo run --example synthesize -- \
    --text "Hello, world." \
    --language en \
    --out hello.wav
```

Model files default to `/opt/vox-models/tts/chatterbox/`. Override with `--t3-gguf` and `--s3gen-gguf`.

## Library Usage

```rust
use chatterbox_rs::{Engine, EngineOptions};

let engine = Engine::new(EngineOptions {
    t3_gguf_path: "models/t3-q4_0.gguf".into(),
    s3gen_gguf_path: "models/s3gen-f16.gguf".into(),
    language: "en".into(),
    ..Default::default()
})?;

let result = engine.synthesize("Hello, world.")?;
// result.pcm: Vec<f32> — mono 24 kHz PCM samples
// result.sample_rate: u32
```

## HTTP/WS Server

```bash
cargo run --example tts_server
curl -X POST -H 'Content-Type: application/json' \
    -d '{"text":"Hello world"}' \
    http://localhost:7860/tts -o hello.wav
```

Routes:
- `POST /tts` — JSON in, WAV binary out
- `GET /tts/stream` — WebSocket: text JSON in, binary PCM chunks out
- `GET /health` — JSON health check

Without the `server` feature, the server module is excluded:
```bash
cargo build --no-default-features
```

## Build

### Prerequisites
- CMake ≥ 3.20
- C++17 compiler
- CUDA Toolkit (autodetected at `/usr/local/cuda`) — optional, CPU fallback

```bash
cargo build --release
```

The vendored `chatterbox-cpp/` (including GGML) is built as static libraries. No `.so` files to ship.

## Tests

```bash
# Unit + integration tests
cargo test

# Verify no-server build
cargo build --no-default-features

# Benchmark comparison vs CLI reference audio
cargo run --example bench_compare
```

Requires model files at `/opt/vox-models/tts/chatterbox/`.

## Comparison with CLI

Benchmark audio from the original `chatterbox.cpp` CLI is in `bench_audio/`. The Rust bindings produce perceptually identical output (GPU vs CPU numerical differences only):

| Language | RMSE  | SNR    | Status |
|----------|-------|--------|--------|
| en       | 0.171 | 15.2 dB | ✅     |
| de       | 0.152 | 14.6 dB | ✅     |
| es       | 0.188 | 13.1 dB | ✅     |
| fr       | 0.182 | 14.6 dB | ✅     |
| it       | 0.177 | 14.9 dB | ✅     |
| ko       | 0.195 | 14.1 dB | ✅     |

## Architecture

```
Rust crate (chatterbox-rs)
  │
  ├── src/
  │   ├── ffi.rs         — extern "C" declarations
  │   ├── engine.rs      — safe Engine wrapper
  │   ├── server.rs      — axum HTTP/WS server
  │   ├── error.rs       — EngineError type
  │   └── lib.rs         — public API
  │
  ├── c_src/
  │   ├── tts_bridge.h   — C bridge header
  │   └── tts_bridge.cpp — extern "C" wrapper
  │
  └── chatterbox-cpp/    — vendored C++ engine (stripped)
      ├── src/           — chatterbox_engine.cpp, mtl_tokenizer, etc.
      ├── include/       — public API headers
      └── ggml/          — GGML tensor library (static)
```

## License

MIT
