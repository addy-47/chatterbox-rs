use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

// FFI Declarations
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CodecArch {
    Unknown = 0,
    WavTokenizerLarge = 1,
    Dac = 2,
    Mimi = 3,
    Qwen3TtsTokenizer = 4,
    Soprano = 5,
    NemoNanoCodec = 6,
    Neucodec = 7,
    DistillNeucodec = 8,
    ChatterboxS3T = 9,
    ChatterboxS3G = 10,
    Xcodec2 = 11,
    Snac = 12,
    MossAudio = 13,
    XyTokenizer = 14,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CodecStatus {
    Success = 0,
    InvalidArg = 1,
    InvalidState = 2,
    IoError = 3,
    NotSupported = 4,
    InternalError = 5,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CodecPcmType {
    F32 = 0,
    I16 = 1,
}

#[repr(C)]
pub struct codec_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct codec_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct codec_lm {
    _private: [u8; 0],
}

#[repr(C)]
pub struct codec_lm_state {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CodecLmInfo {
    pub kind: i32,
    pub hidden_dim: i32,
    pub audio_embed_dim: i32,
    pub compose_audio_embed_dim: i32,
    pub n_codebook: i32,
    pub codebook_sizes: *const i32,
    pub delay_pattern: *const i32,
    pub host_arch: *const c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CodecModelParams {
    pub use_gpu: bool,
    pub n_threads: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CodecContextParams {
    pub seed: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CodecEncodeParams {
    pub n_threads: i32,
    pub frame_size: i32,
    pub hop_size: i32,
    pub n_q: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CodecDecodeParams {
    pub n_threads: i32,
    pub n_q: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CodecAudio {
    pub data: *const c_void,
    pub n_samples: i32,
    pub sample_rate: i32,
    pub n_channels: i32,
    pub pcm_type: CodecPcmType,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CodecTokenBuffer {
    pub data: *mut i32,
    pub n_tokens: i32,
    pub n_frames: i32,
    pub n_q: i32,
    pub codebook_size: i32,
    pub sample_rate: i32,
    pub hop_size: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CodecPcmBuffer {
    pub data: *mut f32,
    pub n_samples: i32,
    pub sample_rate: i32,
    pub n_channels: i32,
}

extern "C" {
    pub fn codec_model_default_params() -> CodecModelParams;
    pub fn codec_context_default_params() -> CodecContextParams;
    pub fn codec_encode_default_params() -> CodecEncodeParams;
    pub fn codec_decode_default_params() -> CodecDecodeParams;

    pub fn codec_model_load_from_file(path_model: *const c_char, params: CodecModelParams) -> *mut codec_model;
    pub fn codec_model_free(model: *mut codec_model);

    pub fn codec_init_from_model(model: *mut codec_model, params: CodecContextParams) -> *mut codec_context;
    pub fn codec_free(ctx: *mut codec_context);

    pub fn codec_encode(
        ctx: *mut codec_context,
        audio: *const CodecAudio,
        out_tokens: *mut CodecTokenBuffer,
        params: CodecEncodeParams,
    ) -> CodecStatus;

    pub fn codec_decode(
        ctx: *mut codec_context,
        tokens: *const CodecTokenBuffer,
        out_pcm: *mut CodecPcmBuffer,
        params: CodecDecodeParams,
    ) -> CodecStatus;

    pub fn codec_token_buffer_free(tokens: *mut CodecTokenBuffer);
    pub fn codec_pcm_buffer_free(pcm: *mut CodecPcmBuffer);

    pub fn codec_get_last_error(ctx: *const codec_context) -> *const c_char;
    pub fn codec_model_arch(model: *const codec_model) -> CodecArch;
    pub fn codec_model_name(model: *const codec_model) -> *const c_char;
    pub fn codec_model_sample_rate(model: *const codec_model) -> i32;
    pub fn codec_model_hop_size(model: *const codec_model) -> i32;

    // codec_lm declarations
    pub fn codec_lm_create(codec: *mut codec_model) -> *mut codec_lm;
    pub fn codec_lm_free(lm: *mut codec_lm);
    pub fn codec_lm_get_info(lm: *const codec_lm) -> *const CodecLmInfo;
    pub fn codec_lm_get_last_error(lm: *const codec_lm) -> *const c_char;
    pub fn codec_lm_state_get_last_error(st: *const codec_lm_state) -> *const c_char;
    pub fn codec_lm_get_create_error() -> *const c_char;

    pub fn codec_lm_state_new(lm: *mut codec_lm) -> *mut codec_lm_state;
    pub fn codec_lm_state_free(st: *mut codec_lm_state);
    pub fn codec_lm_state_reset(st: *mut codec_lm_state);
    pub fn codec_lm_state_set_text_context(st: *mut codec_lm_state, text_token: i32) -> CodecStatus;

    pub fn codec_lm_audio_embd(lm: *mut codec_lm, cb_idx: i32, code: i32) -> *const f32;
    pub fn codec_lm_compose_audio_embd(lm: *mut codec_lm, codes: *const i32, out_embd: *mut f32) -> CodecStatus;

    pub fn codec_lm_step_begin(st: *mut codec_lm_state, h_in: *const f32) -> CodecStatus;
    pub fn codec_lm_step_pending(st: *const codec_lm_state) -> bool;
    pub fn codec_lm_step_logits(st: *mut codec_lm_state, out_cb_idx: *mut i32, out_n: *mut i32) -> *const f32;
    pub fn codec_lm_step_push_code(st: *mut codec_lm_state, code: i32) -> CodecStatus;
    pub fn codec_lm_step_finish(st: *mut codec_lm_state, out_codes: *mut i32) -> CodecStatus;
}

// Safe Rust Wrappers

pub struct CodecModelWrapper {
    ptr: *mut codec_model,
}

impl CodecModelWrapper {
    pub fn load(path: &str, use_gpu: bool, n_threads: i32) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|e| e.to_string())?;
        let mut params = unsafe { codec_model_default_params() };
        params.use_gpu = use_gpu;
        if n_threads > 0 {
            params.n_threads = n_threads;
        }

        let ptr = unsafe { codec_model_load_from_file(c_path.as_ptr(), params) };
        if ptr.is_null() {
            return Err(format!("Failed to load codec model from: {}", path));
        }

        Ok(Self { ptr })
    }

    pub fn name(&self) -> String {
        let name_ptr = unsafe { codec_model_name(self.ptr) };
        if name_ptr.is_null() {
            return "Unknown".to_string();
        }
        unsafe { CStr::from_ptr(name_ptr).to_string_lossy().into_owned() }
    }

    pub fn arch(&self) -> CodecArch {
        unsafe { codec_model_arch(self.ptr) }
    }

    pub fn sample_rate(&self) -> i32 {
        unsafe { codec_model_sample_rate(self.ptr) }
    }

    pub fn hop_size(&self) -> i32 {
        unsafe { codec_model_hop_size(self.ptr) }
    }
}

impl Drop for CodecModelWrapper {
    fn drop(&mut self) {
        unsafe {
            codec_model_free(self.ptr);
        }
    }
}

unsafe impl Send for CodecModelWrapper {}
unsafe impl Sync for CodecModelWrapper {}

pub struct CodecContextWrapper {
    ptr: *mut codec_context,
    _model: std::sync::Arc<CodecModelWrapper>,
}

impl CodecContextWrapper {
    pub fn init(model: std::sync::Arc<CodecModelWrapper>) -> Result<Self, String> {
        let params = unsafe { codec_context_default_params() };
        let ptr = unsafe { codec_init_from_model(model.ptr, params) };
        if ptr.is_null() {
            return Err("Failed to initialize codec context from model".to_string());
        }
        Ok(Self { ptr, _model: model })
    }

    pub fn decode(&self, token_ids: &[i32], n_q: i32) -> Result<Vec<f32>, String> {
        let tokens_buf = CodecTokenBuffer {
            data: token_ids.as_ptr() as *mut i32,
            n_tokens: token_ids.len() as i32,
            n_frames: token_ids.len() as i32,
            n_q,
            codebook_size: 6561,
            sample_rate: 24000,
            hop_size: 960,
        };

        let mut out_pcm = CodecPcmBuffer {
            data: ptr::null_mut(),
            n_samples: 0,
            sample_rate: 0,
            n_channels: 0,
        };

        let mut params = unsafe { codec_decode_default_params() };
        params.n_q = n_q;

        let status = unsafe {
            codec_decode(
                self.ptr,
                &tokens_buf as *const CodecTokenBuffer,
                &mut out_pcm as *mut CodecPcmBuffer,
                params,
            )
        };

        if status != CodecStatus::Success {
            let err_ptr = unsafe { codec_get_last_error(self.ptr) };
            let err_str = if err_ptr.is_null() {
                "Unknown decode error".to_string()
            } else {
                unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
            };
            return Err(err_str);
        }

        // Copy decoded samples to Rust Vec
        let samples = unsafe {
            std::slice::from_raw_parts(out_pcm.data, out_pcm.n_samples as usize).to_vec()
        };

        // Free C-allocated buffer
        unsafe {
            codec_pcm_buffer_free(&mut out_pcm as *mut CodecPcmBuffer);
        }

        Ok(samples)
    }
}

impl Drop for CodecContextWrapper {
    fn drop(&mut self) {
        unsafe {
            codec_free(self.ptr);
        }
    }
}

unsafe impl Send for CodecContextWrapper {}
unsafe impl Sync for CodecContextWrapper {}

// Safe wrappers for codec_lm
pub struct CodecLmWrapper {
    pub ptr: *mut codec_lm,
    _codec: std::sync::Arc<CodecModelWrapper>,
}

impl CodecLmWrapper {
    pub fn create(codec: std::sync::Arc<CodecModelWrapper>) -> Result<Self, String> {
        let ptr = unsafe { codec_lm_create(codec.ptr) };
        if ptr.is_null() {
            let err_ptr = unsafe { codec_lm_get_create_error() };
            let err_str = if err_ptr.is_null() {
                "Unknown codec_lm creation error".to_string()
            } else {
                unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
            };
            return Err(err_str);
        }
        Ok(Self { ptr, _codec: codec })
    }

    pub fn get_info(&self) -> CodecLmInfo {
        unsafe { *codec_lm_get_info(self.ptr) }
    }

    pub fn compose_audio_embd(&self, codes: &[i32], out_embd: &mut [f32]) -> Result<(), String> {
        let status = unsafe { codec_lm_compose_audio_embd(self.ptr, codes.as_ptr(), out_embd.as_mut_ptr()) };
        if status != CodecStatus::Success {
            let err_ptr = unsafe { codec_lm_get_last_error(self.ptr) };
            let err_str = if err_ptr.is_null() {
                "Unknown compose_audio_embd error".to_string()
            } else {
                unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
            };
            return Err(err_str);
        }
        Ok(())
    }
}

impl Drop for CodecLmWrapper {
    fn drop(&mut self) {
        unsafe {
            codec_lm_free(self.ptr);
        }
    }
}

unsafe impl Send for CodecLmWrapper {}
unsafe impl Sync for CodecLmWrapper {}

pub struct CodecLmStateWrapper {
    pub ptr: *mut codec_lm_state,
    _lm: std::sync::Arc<CodecLmWrapper>,
}

impl CodecLmStateWrapper {
    pub fn new(lm: std::sync::Arc<CodecLmWrapper>) -> Result<Self, String> {
        let ptr = unsafe { codec_lm_state_new(lm.ptr) };
        if ptr.is_null() {
            return Err("Failed to create codec_lm state".to_string());
        }
        Ok(Self { ptr, _lm: lm })
    }

    pub fn reset(&self) {
        unsafe { codec_lm_state_reset(self.ptr) };
    }

    pub fn set_text_context(&self, text_token: i32) -> Result<(), String> {
        let status = unsafe { codec_lm_state_set_text_context(self.ptr, text_token) };
        if status != CodecStatus::Success {
            return Err(format!("Failed to set text context: {:?}", status));
        }
        Ok(())
    }

    pub fn step_begin(&self, h_in: &[f32]) -> Result<(), String> {
        let status = unsafe { codec_lm_step_begin(self.ptr, h_in.as_ptr()) };
        if status != CodecStatus::Success {
            let err_ptr = unsafe { codec_lm_state_get_last_error(self.ptr) };
            let err_str = if err_ptr.is_null() {
                "Unknown step_begin error".to_string()
            } else {
                unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
            };
            return Err(err_str);
        }
        Ok(())
    }

    pub fn step_pending(&self) -> bool {
        unsafe { codec_lm_step_pending(self.ptr) }
    }

    pub fn step_logits(&self) -> Result<(i32, &[f32]), String> {
        let mut cb_idx = 0;
        let mut n = 0;
        let ptr = unsafe { codec_lm_step_logits(self.ptr, &mut cb_idx, &mut n) };
        if ptr.is_null() {
            let err_ptr = unsafe { codec_lm_state_get_last_error(self.ptr) };
            let err_str = if err_ptr.is_null() {
                "Unknown step_logits error".to_string()
            } else {
                unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
            };
            return Err(err_str);
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, n as usize) };
        Ok((cb_idx, slice))
    }

    pub fn step_push_code(&self, code: i32) -> Result<(), String> {
        let status = unsafe { codec_lm_step_push_code(self.ptr, code) };
        if status != CodecStatus::Success {
            let err_ptr = unsafe { codec_lm_state_get_last_error(self.ptr) };
            let err_str = if err_ptr.is_null() {
                "Unknown step_push_code error".to_string()
            } else {
                unsafe { CStr::from_ptr(err_ptr).to_string_lossy().into_owned() }
            };
            return Err(err_str);
        }
        Ok(())
    }

    pub fn step_finish(&self, out_codes: &mut [i32]) -> Result<(), String> {
        let status = unsafe { codec_lm_step_finish(self.ptr, out_codes.as_mut_ptr()) };
        if status != CodecStatus::Success {
            return Err(format!("Failed to finish step: {:?}", status));
        }
        Ok(())
    }
}

impl Drop for CodecLmStateWrapper {
    fn drop(&mut self) {
        unsafe {
            codec_lm_state_free(self.ptr);
        }
    }
}

unsafe impl Send for CodecLmStateWrapper {}
unsafe impl Sync for CodecLmStateWrapper {}
