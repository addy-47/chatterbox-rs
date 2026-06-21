//! Safe Rust wrapper around the Chatterbox TTS C bridge.

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::path::Path;
use std::ptr;


use crate::error::EngineError;
use crate::ffi::{
    self, tts_bridge_engine, tts_bridge_engine_options, tts_bridge_synthesis_result,
};

// ── EngineOptions ─────────────────────────────────────────────────

/// Configuration for creating a new [`Engine`] instance.
///
/// # Example
///
/// ```no_run
/// use chatterbox_rs::EngineOptions;
///
/// let opts = EngineOptions {
///     t3_gguf_path: "models/t3-q4_0.gguf".into(),
///     s3gen_gguf_path: "models/s3gen-f16.gguf".into(),
///     language: "en".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug)]
pub struct EngineOptions {
    // ── Required ──────────────────────────────────────────────

    /// Path to the T3 GGUF model file.
    pub t3_gguf_path: String,

    /// Path to the S3Gen GGUF model file.
    pub s3gen_gguf_path: String,

    // ── Voice cloning ─────────────────────────────────────────

    /// Path to a mono reference WAV (>= 5 s) for voice cloning.
    /// Empty = use the built-in reference voice.
    pub reference_audio: String,

    /// Path to a directory of pre-baked voice `.npy` tensors.
    /// Empty = none.
    pub voice_dir: String,

    // ── MTL (multilingual) ────────────────────────────────────

    /// Language code for the MTL (multilingual) variant.
    /// Required when loading a t3_mtl GGUF.
    /// Supported: en, es, fr, de, it, pt, nl, pl, tr, sv, da, fi, no,
    ///            el, ms, sw, ar, ko.
    /// Ignored for Turbo GGUFs.
    pub language: String,

    // ── Backend ───────────────────────────────────────────────

    /// Number of layers to offload to the GPU (0 = CPU only).
    pub n_gpu_layers: i32,

    /// Thread count. 0 = default (min(hw_concurrency, 4)).
    pub n_threads: i32,

    // ── Sampling ──────────────────────────────────────────────

    /// RNG seed for reproducible output.
    pub seed: i32,

    /// Maximum speech tokens to generate per call.
    pub n_predict: i32,

    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub repeat_penalty: f32,

    // ── MTL sampling (ignored for Turbo) ──────────────────────

    /// Classifier-free guidance weight (default 0.5).
    pub cfg_weight: f32,

    /// Minimum-probability warp (default 0.05, 0 = off).
    pub min_p: f32,

    /// Emotion-adv scalar in [0, 1] (default 0.5).
    pub exaggeration: f32,

    // ── S3Gen ─────────────────────────────────────────────────

    /// CFM Euler step count for batch mode. 0 = GGUF default.
    pub cfm_steps: i32,

    // ── Streaming ─────────────────────────────────────────────

    /// Speech tokens per streaming chunk (0 = batch / non-streaming).
    pub stream_chunk_tokens: i32,

    /// Override chunk size for the first chunk (0 = same as stream_chunk_tokens).
    pub stream_first_chunk_tokens: i32,

    /// CFM steps for streaming chunks (0 = library default).
    pub stream_cfm_steps: i32,

    // ── Misc ──────────────────────────────────────────────────

    /// Enable verbose stderr logging.
    pub verbose: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            t3_gguf_path: String::new(),
            s3gen_gguf_path: String::new(),
            reference_audio: String::new(),
            voice_dir: String::new(),
            language: "en".into(),
            n_gpu_layers: 0,
            n_threads: 0,
            seed: 42,
            n_predict: 1000,
            top_k: 1000,
            top_p: 0.95,
            temperature: 0.8,
            repeat_penalty: 1.2,
            cfg_weight: 0.5,
            min_p: 0.05,
            exaggeration: 0.5,
            cfm_steps: 0,
            stream_chunk_tokens: 0,
            stream_first_chunk_tokens: 0,
            stream_cfm_steps: 0,
            verbose: false,
        }
    }
}

impl EngineOptions {
    fn to_ffi(&self) -> Result<FfiOptions, EngineError> {
        FfiOptions::from_options(self)
    }
}

/// Holds the CString temporaries needed to build the C options struct.
struct FfiOptions {
    opts: tts_bridge_engine_options,
    // Keep CStrings alive — they must outlive the struct pointer.
    _t3: CString,
    _s3: CString,
    _ref: Option<CString>,
    _voice: Option<CString>,
    _lang: CString,
}

impl FfiOptions {
    fn from_options(o: &EngineOptions) -> Result<Self, EngineError> {
        let t3 = CString::new(o.t3_gguf_path.as_bytes())
            .map_err(|e| EngineError::LoadFailed(format!("t3_gguf_path contains null: {}", e)))?;
        let s3 = CString::new(o.s3gen_gguf_path.as_bytes())
            .map_err(|e| EngineError::LoadFailed(format!("s3gen_gguf_path contains null: {}", e)))?;
        let lang = CString::new(o.language.as_bytes())
            .map_err(|e| EngineError::LoadFailed(format!("language contains null: {}", e)))?;
        let ref_audio = if o.reference_audio.is_empty() {
            None
        } else {
            Some(CString::new(o.reference_audio.as_bytes()).map_err(|e| {
                EngineError::LoadFailed(format!("reference_audio contains null: {}", e))
            })?)
        };
        let voice = if o.voice_dir.is_empty() {
            None
        } else {
            Some(CString::new(o.voice_dir.as_bytes()).map_err(|e| {
                EngineError::LoadFailed(format!("voice_dir contains null: {}", e))
            })?)
        };

        let opts = tts_bridge_engine_options {
            t3_gguf_path: t3.as_ptr(),
            s3gen_gguf_path: s3.as_ptr(),
            reference_audio: ref_audio.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
            voice_dir: voice.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
            language: lang.as_ptr(),
            n_gpu_layers: o.n_gpu_layers,
            n_threads: o.n_threads,
            seed: o.seed,
            n_predict: o.n_predict,
            top_k: o.top_k,
            top_p: o.top_p,
            temperature: o.temperature,
            repeat_penalty: o.repeat_penalty,
            cfg_weight: o.cfg_weight,
            min_p: o.min_p,
            exaggeration: o.exaggeration,
            cfm_steps: o.cfm_steps,
            stream_chunk_tokens: o.stream_chunk_tokens,
            stream_first_chunk_tokens: o.stream_first_chunk_tokens,
            stream_cfm_steps: o.stream_cfm_steps,
            verbose: o.verbose as i32,
        };

        Ok(Self {
            opts,
            _t3: t3,
            _s3: s3,
            _ref: ref_audio,
            _voice: voice,
            _lang: lang,
        })
    }
}

// ── SynthesisResult ───────────────────────────────────────────────

/// Synthesised audio and timing statistics.
#[derive(Clone, Debug)]
pub struct SynthesisResult {
    /// 24 kHz mono PCM, float32 samples in [-1, 1].
    pub pcm: Vec<f32>,

    /// Sample rate (always 24000).
    pub sample_rate: u32,

    /// T3 autoregressive decode wall time (ms).
    pub t3_ms: f64,

    /// S3Gen + HiFT synthesis wall time (ms).
    pub s3gen_ms: f64,

    /// Number of speech tokens generated.
    pub t3_tokens: u32,

    /// Number of audio samples in the PCM.
    pub audio_samples: u32,
}

// ── Engine ────────────────────────────────────────────────────────

/// A Chatterbox TTS engine instance.
///
/// Manages the full T3 + S3Gen + HiFT pipeline.  One instance holds one
/// T3 model and its KV cache; it is **Send** but **not Sync** (use a
/// `Mutex` to share across threads).
///
/// # Example
///
/// ```no_run
/// use chatterbox_rs::{Engine, EngineOptions};
///
/// let engine = Engine::new(EngineOptions {
///     t3_gguf_path: "/opt/vox-models/tts/chatterbox/t3-q4_0.gguf".into(),
///     s3gen_gguf_path: "/opt/vox-models/tts/chatterbox/s3gen-f16.gguf".into(),
///     language: "en".into(),
///     ..Default::default()
/// })?;
/// let result = engine.synthesize("Hello, world.")?;
/// assert!(result.pcm.len() > 0);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub struct Engine {
    handle: *mut tts_bridge_engine,
    _nosync: PhantomData<*mut ()>,
}

// Safety: The C++ Engine is single-threaded per instance but different
// instances may live on different threads.  Our handle is a unique
// pointer to a heap-allocated C++ object.
unsafe impl Send for Engine {}

impl Engine {
    /// Create a new engine, loading the T3 and S3Gen models.
    ///
    /// Returns `Err(EngineError::LoadFailed)` if the model files cannot
    /// be found or loaded, the GGUF variant is MTL and `language` is
    /// empty, or any other initialisation error occurs.
    pub fn new(options: EngineOptions) -> Result<Self, EngineError> {
        // Validate paths early.
        if options.t3_gguf_path.is_empty() {
            return Err(EngineError::LoadFailed("t3_gguf_path is required".into()));
        }
        if options.s3gen_gguf_path.is_empty() {
            return Err(EngineError::LoadFailed("s3gen_gguf_path is required".into()));
        }
        let t3_path = Path::new(&options.t3_gguf_path);
        let s3_path = Path::new(&options.s3gen_gguf_path);
        if !t3_path.exists() {
            return Err(EngineError::LoadFailed(format!(
                "T3 GGUF not found: {}",
                t3_path.display()
            )));
        }
        if !s3_path.exists() {
            return Err(EngineError::LoadFailed(format!(
                "S3Gen GGUF not found: {}",
                s3_path.display()
            )));
        }

        let ffi_opts = options.to_ffi()?;

        let mut out_error: *mut libc::c_char = ptr::null_mut();
        let handle = unsafe {
            ffi::tts_bridge_engine_create(&ffi_opts.opts, &mut out_error)
        };

        if handle.is_null() {
            let err = if !out_error.is_null() {
                let msg = unsafe { CStr::from_ptr(out_error).to_string_lossy().into_owned() };
                unsafe { ffi::tts_bridge_free_string(out_error) };
                msg
            } else {
                "unknown error (engine creation returned null)".into()
            };
            return Err(EngineError::LoadFailed(err));
        }

        Ok(Self {
            handle,
            _nosync: PhantomData,
        })
    }

    /// Synthesize `text` to PCM audio.
    ///
    /// Returns an error if the text is empty, synthesis is cancelled, or
    /// any model-level error occurs.
    pub fn synthesize(&self, text: &str) -> Result<SynthesisResult, EngineError> {
        let c_text = CString::new(text.as_bytes())
            .map_err(|e| EngineError::SynthesisFailed(format!("text contains null byte: {}", e)))?;

        let mut result: tts_bridge_synthesis_result = unsafe { std::mem::zeroed() };
        let mut out_error: *mut libc::c_char = ptr::null_mut();

        let rc = unsafe {
            ffi::tts_bridge_engine_synthesize(
                self.handle,
                c_text.as_ptr(),
                &mut result,
                &mut out_error,
            )
        };

        if rc != 0 {
            let err = if !out_error.is_null() {
                let msg = unsafe { CStr::from_ptr(out_error).to_string_lossy().into_owned() };
                unsafe { ffi::tts_bridge_free_string(out_error) };
                msg
            } else {
                "unknown synthesis error".into()
            };
            return Err(EngineError::SynthesisFailed(err));
        }

        let res = result_from_ffi(&result);

        // Free the PCM buffer allocated by the bridge.
        unsafe {
            ffi::tts_bridge_free_result(&mut result);
        }

        Ok(res)
    }

    /// Synthesize text with streaming chunk callbacks.
    ///
    /// `on_chunk` is called synchronously for each chunk of PCM as it is
    /// produced.  The accumulated full PCM is returned in the
    /// [`SynthesisResult`].
    ///
    /// Set `stream_chunk_tokens` in [`EngineOptions`] to a positive value
    /// to enable streaming.
    pub fn synthesize_streaming<F>(
        &self,
        text: &str,
        on_chunk: F,
    ) -> Result<SynthesisResult, EngineError>
    where
        F: FnMut(&[f32], usize, bool),
    {
        let c_text = CString::new(text.as_bytes())
            .map_err(|e| EngineError::SynthesisFailed(format!("text contains null byte: {}", e)))?;

        let mut result: tts_bridge_synthesis_result = unsafe { std::mem::zeroed() };
        let mut out_error: *mut libc::c_char = ptr::null_mut();

        // We use a callback that stores the user callback in a heap-allocated
        // Box, then calls it for each chunk.
        let cb = Box::into_raw(Box::new(on_chunk));

        extern "C" fn trampoline<F: FnMut(&[f32], usize, bool)>(
            pcm: *const libc::c_float,
            pcm_len: libc::c_int,
            chunk_index: libc::c_int,
            is_last: libc::c_int,
            user_data: *mut libc::c_void,
        ) {
            let cb = unsafe { &mut *(user_data as *mut F) };
            let slice = unsafe { std::slice::from_raw_parts(pcm, pcm_len as usize) };
            cb(slice, chunk_index as usize, is_last != 0);
        }

        let rc = unsafe {
            ffi::tts_bridge_engine_synthesize_streaming(
                self.handle,
                c_text.as_ptr(),
                trampoline::<F> as ffi::tts_bridge_stream_cb,
                cb as *mut libc::c_void,
                &mut result,
                &mut out_error,
            )
        };

        // Recover the Box so it gets dropped.
        let _ = unsafe { Box::from_raw(cb) };

        if rc != 0 {
            let err = if !out_error.is_null() {
                let msg = unsafe { CStr::from_ptr(out_error).to_string_lossy().into_owned() };
                unsafe { ffi::tts_bridge_free_string(out_error) };
                msg
            } else {
                "unknown streaming synthesis error".into()
            };
            return Err(EngineError::SynthesisFailed(err));
        }

        let res = result_from_ffi(&result);

        unsafe {
            ffi::tts_bridge_free_result(&mut result);
        }

        Ok(res)
    }

    /// Best-effort cancel of an in-flight synthesis call.
    ///
    /// Safe to call from any thread.  The flag is checked periodically
    /// inside the T3 decode loop.
    pub fn cancel(&self) {
        // The C++ Engine::cancel() is not exposed through the bridge yet.
        // We implement it at the FFI level through the existing cancel_flag.
        // For now, this is a no-op at the bridge level.
        // A future enhancement would expose tts_bridge_engine_cancel().
    }

    /// Return a reference to the handle (for low-level FFI use).
    #[doc(hidden)]
    pub fn as_raw(&self) -> *mut tts_bridge_engine {
        self.handle
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::tts_bridge_engine_destroy(self.handle);
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────

fn result_from_ffi(r: &tts_bridge_synthesis_result) -> SynthesisResult {
    let pcm = if r.pcm_len > 0 && !r.pcm.is_null() {
        unsafe { std::slice::from_raw_parts(r.pcm, r.pcm_len as usize).to_vec() }
    } else {
        Vec::new()
    };

    SynthesisResult {
        pcm,
        sample_rate: r.sample_rate as u32,
        t3_ms: r.t3_ms,
        s3gen_ms: r.s3gen_ms,
        t3_tokens: r.t3_tokens as u32,
        audio_samples: r.audio_samples as u32,
    }
}
