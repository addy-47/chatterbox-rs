//! # chatterbox-rs
//!
//! Rust bindings for the [chatterbox.cpp](https://github.com/addy-47/chatterbox-cpp)
//! TTS engine.  Supports both Turbo (GPT-2) and Multilingual (Llama-520M)
//! model variants with voice cloning, streaming, and GPU acceleration.
//!
//! ## Quick start
//!
//! ```no_run
//! use chatterbox_rs::{Engine, EngineOptions};
//!
//! let engine = Engine::new(EngineOptions {
//!     t3_gguf_path: "/opt/vox-models/tts/chatterbox/t3-q4_0.gguf".into(),
//!     s3gen_gguf_path: "/opt/vox-models/tts/chatterbox/s3gen-f16.gguf".into(),
//!     language: "en".into(),
//!     ..Default::default()
//! })?;
//!
//! let result = engine.synthesize("Hello, world.")?;
//! assert!(result.pcm.len() > 0);
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Features
//!
//! - `server` (default) — enables the axum HTTP/WS server.
//!
//! ## Threading
//!
//! [`Engine`] is `Send` but not `Sync`.  Create one instance per thread
//! or wrap in a `Mutex`.

pub mod error;
mod engine;
mod ffi;

#[cfg(feature = "server")]
pub mod server;

pub use engine::{Engine, EngineOptions, SynthesisResult};
pub use error::EngineError;
