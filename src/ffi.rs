//! Raw FFI declarations for the Chatterbox TTS C bridge.
//!
//! These mirror the types and functions in `c_src/tts_bridge.h`.
//! They are `unsafe` and should not be called directly — use
//! the safe wrappers in [`crate::Engine`] instead.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_char, c_double, c_float, c_int, c_void};

// ── Opaque handle ────────────────────────────────────────────────

pub enum tts_bridge_engine {}

// ── Options struct (mirrors tts_bridge_engine_options in C) ───────

#[repr(C)]
pub struct tts_bridge_engine_options {
    pub t3_gguf_path: *const c_char,
    pub s3gen_gguf_path: *const c_char,
    pub reference_audio: *const c_char,
    pub voice_dir: *const c_char,
    pub language: *const c_char,

    pub n_gpu_layers: c_int,
    pub n_threads: c_int,
    pub seed: c_int,
    pub n_predict: c_int,

    pub top_k: c_int,
    pub top_p: c_float,
    pub temperature: c_float,
    pub repeat_penalty: c_float,

    pub cfg_weight: c_float,
    pub min_p: c_float,
    pub exaggeration: c_float,

    pub cfm_steps: c_int,

    pub stream_chunk_tokens: c_int,
    pub stream_first_chunk_tokens: c_int,
    pub stream_cfm_steps: c_int,

    pub verbose: c_int,
}

// ── Result struct ─────────────────────────────────────────────────

#[repr(C)]
pub struct tts_bridge_synthesis_result {
    pub pcm: *const c_float,
    pub pcm_len: c_int,
    pub sample_rate: c_int,
    pub t3_ms: c_double,
    pub s3gen_ms: c_double,
    pub t3_tokens: c_int,
    pub audio_samples: c_int,
}

// ── Stream callback type ──────────────────────────────────────────

pub type tts_bridge_stream_cb = unsafe extern "C" fn(
    pcm: *const c_float,
    pcm_len: c_int,
    chunk_index: c_int,
    is_last: c_int,
    user_data: *mut c_void,
);

// ── FFI function declarations ─────────────────────────────────────

extern "C" {
    pub fn tts_bridge_engine_create(
        options: *const tts_bridge_engine_options,
        out_error: *mut *mut c_char,
    ) -> *mut tts_bridge_engine;

    pub fn tts_bridge_engine_destroy(engine: *mut tts_bridge_engine);

    pub fn tts_bridge_engine_synthesize(
        engine: *mut tts_bridge_engine,
        text: *const c_char,
        out_result: *mut tts_bridge_synthesis_result,
        out_error: *mut *mut c_char,
    ) -> c_int;

    pub fn tts_bridge_engine_synthesize_streaming(
        engine: *mut tts_bridge_engine,
        text: *const c_char,
        callback: tts_bridge_stream_cb,
        user_data: *mut c_void,
        out_result: *mut tts_bridge_synthesis_result,
        out_error: *mut *mut c_char,
    ) -> c_int;

    pub fn tts_bridge_free_result(result: *mut tts_bridge_synthesis_result);

    pub fn tts_bridge_free_string(s: *mut c_char);
}
