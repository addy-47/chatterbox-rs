// C-compatible bridge for the Chatterbox TTS Engine.
//
// Compile with the tts-cpp library includes, link against libtts-cpp.a
// (plus the GGML static archives).  See ../build.rs for the build recipe.

#include "tts_bridge.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#include "tts-cpp/chatterbox/engine.h"

namespace cb = tts_cpp::chatterbox;

// ── Internal helpers ──────────────────────────────────────────────

static char * dup_error(const std::exception & e) {
    // strdup is POSIX; on non-POSIX platforms use malloc + memcpy.
    const char * msg = e.what();
    size_t len = std::strlen(msg);
    char * dup = static_cast<char *>(std::malloc(len + 1));
    if (dup) {
        std::memcpy(dup, msg, len + 1);
    }
    return dup;
}

// ── Engine lifecycle ──────────────────────────────────────────────

tts_bridge_engine * tts_bridge_engine_create(
    const tts_bridge_engine_options * opts,
    char ** out_error)
{
    if (!opts) {
        if (out_error) *out_error = dup_error(std::runtime_error("options is NULL"));
        return nullptr;
    }
    if (!opts->t3_gguf_path || !opts->s3gen_gguf_path) {
        if (out_error) *out_error = dup_error(std::runtime_error(
            "t3_gguf_path and s3gen_gguf_path are required"));
        return nullptr;
    }

    try {
        cb::EngineOptions eopts;

        eopts.t3_gguf_path    = opts->t3_gguf_path;
        eopts.s3gen_gguf_path = opts->s3gen_gguf_path;
        eopts.reference_audio = opts->reference_audio ? opts->reference_audio : "";
        eopts.voice_dir       = opts->voice_dir       ? opts->voice_dir       : "";
        eopts.language        = opts->language        ? opts->language        : "en";

        eopts.n_gpu_layers    = opts->n_gpu_layers;
        eopts.n_threads       = opts->n_threads;
        eopts.seed            = opts->seed;
        eopts.n_predict       = opts->n_predict;

        eopts.top_k           = opts->top_k;
        eopts.top_p           = opts->top_p;
        eopts.temperature     = opts->temperature;
        eopts.repeat_penalty  = opts->repeat_penalty;

        eopts.cfg_weight      = opts->cfg_weight;
        eopts.min_p           = opts->min_p;
        eopts.exaggeration    = opts->exaggeration;

        eopts.cfm_steps       = opts->cfm_steps;

        eopts.stream_chunk_tokens       = opts->stream_chunk_tokens;
        eopts.stream_first_chunk_tokens = opts->stream_first_chunk_tokens;
        eopts.stream_cfm_steps          = opts->stream_cfm_steps;

        eopts.verbose = (opts->verbose != 0);

        auto * engine = new cb::Engine(eopts);
        return reinterpret_cast<tts_bridge_engine *>(engine);

    } catch (const std::exception & e) {
        if (out_error) *out_error = dup_error(e);
        return nullptr;
    }
}

void tts_bridge_engine_destroy(tts_bridge_engine * engine) {
    if (!engine) return;
    auto * e = reinterpret_cast<cb::Engine *>(engine);
    delete e;
}

// ── Synthesis ─────────────────────────────────────────────────────

int tts_bridge_engine_synthesize(
    tts_bridge_engine * engine,
    const char * text,
    tts_bridge_synthesis_result * out_result,
    char ** out_error)
{
    if (!engine || !text || !out_result) {
        if (out_error) *out_error = dup_error(std::runtime_error("NULL argument"));
        return -1;
    }

    try {
        auto * e = reinterpret_cast<cb::Engine *>(engine);
        auto result = e->synthesize(text);

        // Copy PCM to a heap buffer the Rust side owns.
        size_t nbytes = result.pcm.size() * sizeof(float);
        float * pcm_copy = static_cast<float *>(std::malloc(nbytes));
        if (!pcm_copy && !result.pcm.empty()) {
            throw std::bad_alloc();
        }
        std::memcpy(pcm_copy, result.pcm.data(), nbytes);

        out_result->pcm          = pcm_copy;
        out_result->pcm_len      = static_cast<int>(result.pcm.size());
        out_result->sample_rate  = result.sample_rate;
        out_result->t3_ms        = result.t3_ms;
        out_result->s3gen_ms     = result.s3gen_ms;
        out_result->t3_tokens    = result.t3_tokens;
        out_result->audio_samples = result.audio_samples;

        return 0;

    } catch (const std::exception & e) {
        if (out_error) *out_error = dup_error(e);
        return -1;
    }
}

int tts_bridge_engine_synthesize_streaming(
    tts_bridge_engine * engine,
    const char * text,
    tts_bridge_stream_cb callback,
    void * user_data,
    tts_bridge_synthesis_result * out_result,
    char ** out_error)
{
    if (!engine || !text || !out_result) {
        if (out_error) *out_error = dup_error(std::runtime_error("NULL argument"));
        return -1;
    }

    try {
        auto * e = reinterpret_cast<cb::Engine *>(engine);

        // Build the C++ stream callback that delegates to the C callback.
        cb::StreamCallback cpp_cb;
        if (callback) {
            cpp_cb = [callback, user_data](
                const float * pcm, std::size_t samples,
                int chunk_index, bool is_last)
            {
                callback(pcm, static_cast<int>(samples),
                         chunk_index, is_last ? 1 : 0, user_data);
            };
        }

        auto result = e->synthesize(text, cpp_cb);

        size_t nbytes = result.pcm.size() * sizeof(float);
        float * pcm_copy = static_cast<float *>(std::malloc(nbytes));
        if (!pcm_copy && !result.pcm.empty()) {
            throw std::bad_alloc();
        }
        std::memcpy(pcm_copy, result.pcm.data(), nbytes);

        out_result->pcm          = pcm_copy;
        out_result->pcm_len      = static_cast<int>(result.pcm.size());
        out_result->sample_rate  = result.sample_rate;
        out_result->t3_ms        = result.t3_ms;
        out_result->s3gen_ms     = result.s3gen_ms;
        out_result->t3_tokens    = result.t3_tokens;
        out_result->audio_samples = result.audio_samples;

        return 0;

    } catch (const std::exception & e) {
        if (out_error) *out_error = dup_error(e);
        return -1;
    }
}

// ── Memory management ─────────────────────────────────────────────

void tts_bridge_free_result(tts_bridge_synthesis_result * result) {
    if (!result) return;
    // pcm was allocated with std::malloc in the synthesize functions above.
    std::free(const_cast<float *>(result->pcm));
    result->pcm     = nullptr;
    result->pcm_len = 0;
}

void tts_bridge_free_string(char * error_str) {
    std::free(error_str);
}
