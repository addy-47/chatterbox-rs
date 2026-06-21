#pragma once

// C-compatible bridge for the Chatterbox TTS Engine.
// This header can be consumed from both C++ (bridge impl) and
// Rust FFI (via extern "C" declarations).

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a Chatterbox engine instance.
typedef struct tts_bridge_engine tts_bridge_engine;

// Options for creating a new engine.  Must be zero-initialised then
// filled; use the designated-initialiser pattern in C++ or memset + field
// assignment in C.  String pointers must remain valid for the duration of
// the tts_bridge_engine_create call (they are copied internally).
typedef struct {
    const char * t3_gguf_path;
    const char * s3gen_gguf_path;
    const char * reference_audio;   // NULL = use built-in
    const char * voice_dir;         // NULL = none
    const char * language;          // MTL language code, NULL = "en"

    int   n_gpu_layers;
    int   n_threads;                // 0 = default (min(hw_concurrency, 4))
    int   seed;                     // RNG seed for reproducibility
    int   n_predict;                // Max speech tokens per call

    // Sampling knobs.
    int   top_k;
    float top_p;
    float temperature;
    float repeat_penalty;

    // MTL-specific (ignored for Turbo).
    float cfg_weight;
    float min_p;
    float exaggeration;

    // S3Gen batch mode.  0 = GGUF default.
    int   cfm_steps;

    // Streaming chunk tokens.  0 = batch mode.
    int   stream_chunk_tokens;
    int   stream_first_chunk_tokens;
    int   stream_cfm_steps;

    int   verbose;                  // non-zero = verbose stderr logging
} tts_bridge_engine_options;

// Synthesis result.  `pcm` is heap-allocated and must be freed via
// tts_bridge_free_result().  All other fields are valid for the
// lifetime of the result struct.
typedef struct {
    const float * pcm;          // 24 kHz mono f32 PCM, heap-allocated
    int           pcm_len;      // number of float samples
    int           sample_rate;  // always 24000

    // Timing / stats.
    double t3_ms;
    double s3gen_ms;
    int    t3_tokens;
    int    audio_samples;
} tts_bridge_synthesis_result;

// Stream callback signature.
//   pcm          pointer to the chunk's PCM (valid only during callback)
//   pcm_len      number of float samples in this chunk
//   chunk_index  0-based chunk index within the utterance
//   is_last      non-zero on the final chunk
//   user_data    opaque pointer passed to synthesize_streaming
typedef void (*tts_bridge_stream_cb)(const float * pcm, int pcm_len,
                                     int chunk_index, int is_last,
                                     void * user_data);

// ── Engine lifecycle ──────────────────────────────────────────────

// Create a new engine.  Returns a non-NULL handle on success.
// On failure returns NULL and sets *out_error to a malloc'd string
// (caller must free with tts_bridge_free_string).
tts_bridge_engine * tts_bridge_engine_create(
    const tts_bridge_engine_options * options,
    char ** out_error);

// Destroy an engine.  Safe to call with NULL.
void tts_bridge_engine_destroy(tts_bridge_engine * engine);

// ── Synthesis (batch) ─────────────────────────────────────────────

// Synthesise `text` into PCM audio.  Returns 0 on success, -1 on error.
// On error *out_error is set (malloc'd, free via tts_bridge_free_string).
// On success *out_result is filled and must be freed via
// tts_bridge_free_result.
int tts_bridge_engine_synthesize(
    tts_bridge_engine * engine,
    const char * text,
    tts_bridge_synthesis_result * out_result,
    char ** out_error);

// ── Synthesis (streaming) ─────────────────────────────────────────

// Streaming variant: `callback` is invoked synchronously for each chunk
// as it is produced.  The accumulated full PCM is also returned in
// *out_result (which must still be freed).
int tts_bridge_engine_synthesize_streaming(
    tts_bridge_engine * engine,
    const char * text,
    tts_bridge_stream_cb callback,
    void * user_data,
    tts_bridge_synthesis_result * out_result,
    char ** out_error);

// ── Memory management ─────────────────────────────────────────────

void tts_bridge_free_result(tts_bridge_synthesis_result * result);
void tts_bridge_free_string(char * error_str);

#ifdef __cplusplus
} // extern "C"
#endif
