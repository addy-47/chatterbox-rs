use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::{Arc, Mutex};
use std::path::{Path, PathBuf};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use futures_util::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use chatterbox_rs::{
    CodecModelWrapper, CodecContextWrapper, CodecLmWrapper, CodecLmStateWrapper, CodecArch
};


// ---------------------------------------------------------------------
// LlamaBackbone C FFI Wrapper
// ---------------------------------------------------------------------
pub struct LlamaBackbone {
    model: *mut llama_cpp_sys_4::llama_model,
    ctx: *mut llama_cpp_sys_4::llama_context,
    hidden: usize,
    pos: i32,
    n_ctx: i32,
}

unsafe impl Send for LlamaBackbone {}
unsafe impl Sync for LlamaBackbone {}

impl LlamaBackbone {
    pub fn load(gguf_path: &str, n_ctx: i32, use_gpu: bool) -> Result<Self, String> {
        unsafe {
            llama_cpp_sys_4::llama_backend_init();

            let mut mp = llama_cpp_sys_4::llama_model_default_params();
            mp.use_mmap = true;
            if use_gpu {
                mp.n_gpu_layers = 99; // offload all layers to GPU
            } else {
                mp.n_gpu_layers = 0;
            }

            let c_path = CString::new(gguf_path).map_err(|e| e.to_string())?;
            let model = llama_cpp_sys_4::llama_model_load_from_file(c_path.as_ptr(), mp);
            if model.is_null() {
                return Err(format!("Failed to load llama model from: {}", gguf_path));
            }

            let hidden = llama_cpp_sys_4::llama_model_n_embd(model) as usize;

            let mut cp = llama_cpp_sys_4::llama_context_default_params();
            cp.n_ctx = n_ctx as u32;
            cp.n_batch = std::cmp::max(n_ctx, 64) as u32;
            cp.n_ubatch = std::cmp::max(n_ctx, 64) as u32;
            cp.embeddings = true;
            cp.pooling_type = llama_cpp_sys_4::LLAMA_POOLING_TYPE_NONE;

            let ctx = llama_cpp_sys_4::llama_init_from_model(model, cp);

            if ctx.is_null() {
                llama_cpp_sys_4::llama_model_free(model);
                return Err("Failed to initialize llama context".to_string());
            }

            Ok(Self {
                model,
                ctx,
                hidden,
                pos: 0,
                n_ctx,
            })
        }
    }

    pub fn feed_embeds(&mut self, input_embeds: &[f32]) -> Result<Vec<f32>, String> {
        let t_tokens = (input_embeds.len() / self.hidden) as i32;
        if self.pos + t_tokens > self.n_ctx {
            return Err(format!(
                "Backbone context exhausted: pos={} + T={} > n_ctx={}",
                self.pos, t_tokens, self.n_ctx
            ));
        }

        unsafe {
            let mut batch = llama_cpp_sys_4::llama_batch_init(t_tokens, self.hidden as i32, 1);
            batch.n_tokens = t_tokens;

            // Copy input embeds into the batch.embd buffer
            std::ptr::copy_nonoverlapping(
                input_embeds.as_ptr(),
                batch.embd,
                input_embeds.len(),
            );

            for t in 0..t_tokens {
                *batch.pos.add(t as usize) = (self.pos + t) as llama_cpp_sys_4::llama_pos;
                *batch.n_seq_id.add(t as usize) = 1;
                let seq_ptr = *batch.seq_id.add(t as usize);
                *seq_ptr = 0;
                *batch.logits.add(t as usize) = if t == t_tokens - 1 { 1 } else { 0 };
            }

            let rc = llama_cpp_sys_4::llama_decode(self.ctx, batch);
            if rc != 0 {
                llama_cpp_sys_4::llama_batch_free(batch);
                return Err(format!("llama_decode failed with code {}", rc));
            }

            // Fetch embeddings at the only position where logits=1 (index -1, the last embedding)
            let emb_ptr = llama_cpp_sys_4::llama_get_embeddings_ith(self.ctx, -1);
            if emb_ptr.is_null() {
                llama_cpp_sys_4::llama_batch_free(batch);
                return Err("llama_get_embeddings_ith returned NULL".to_string());
            }

            let out_emb = std::slice::from_raw_parts(emb_ptr, self.hidden).to_vec();

            llama_cpp_sys_4::llama_batch_free(batch);
            self.pos += t_tokens;

            Ok(out_emb)
        }
    }

    pub fn clear_kv_cache(&mut self) {
        unsafe {
            let mem = llama_cpp_sys_4::llama_get_memory(self.ctx);
            llama_cpp_sys_4::llama_memory_clear(mem, true);
        }
        self.pos = 0;
    }
}

impl Drop for LlamaBackbone {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                llama_cpp_sys_4::llama_free(self.ctx);
            }
            if !self.model.is_null() {
                llama_cpp_sys_4::llama_model_free(self.model);
            }
        }
    }
}

// ---------------------------------------------------------------------
// Server State and Config
// ---------------------------------------------------------------------
struct AppStateInner {
    s3g_model: Option<Arc<CodecModelWrapper>>,
    s3g_ctx: Option<Arc<CodecContextWrapper>>,
    s3g_lm: Option<Arc<CodecLmWrapper>>,
    tokenizer: Option<Tokenizer>,
    loaded_llama_quant: Option<String>,
    // Llama backbones for conditional / unconditional CFG
    llama_cond: Option<Arc<Mutex<LlamaBackbone>>>,
    llama_uncond: Option<Arc<Mutex<LlamaBackbone>>>,
    cond_emb: Option<Vec<f32>>,
    text_emb: Option<Vec<f32>>,
    text_pos_emb: Option<Vec<f32>>,
    speech_emb: Option<Vec<f32>>,
    speech_pos_emb: Option<Vec<f32>>,
    speech_head: Option<Vec<f32>>,
}

type SharedState = Arc<Mutex<AppStateInner>>;

#[derive(Serialize)]
struct HealthResponse {
    backend: String,
    rtf_estimate: f32,
    model_loaded: bool,
    vram_mb: u32,
    gpu_name: String,
}

#[derive(Deserialize)]
struct SetupRequest {
    llama_quant: String, // "q4", "q8", or "fp16"
}

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
    llama_quant: Option<String>,
    exaggeration: Option<f32>,
    cfg_weight: Option<f32>,
    temperature: Option<f32>,
}

// Custom simple Rng to avoid adding rand to Cargo.toml
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state as f64) / (u64::MAX as f64)
    }
}

// ---------------------------------------------------------------------
// Helper text normalization (matching Python EnTokenizer / Punctuation Norm)
// ---------------------------------------------------------------------
fn chatterbox_punc_norm(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }
    
    // Capitalize first char if lower
    let mut chars = text.chars();
    let mut normalized = match chars.next() {
        None => String::new(),
        Some(c) => {
            if c.is_lowercase() {
                c.to_uppercase().collect::<String>() + chars.as_str()
            } else {
                text.to_string()
            }
        }
    };
    
    // Split and join to normalize whitespace
    normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    
    // Replacements
    let replacements = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ];
    
    for (old, new) in replacements.iter() {
        normalized = normalized.replace(old, new);
    }
    
    normalized = normalized.trim_end().to_string();
    
    // Sentence ender check
    let sentence_enders = ['.', '!', '?', '-', ','];
    if let Some(last_char) = normalized.chars().last() {
        if !sentence_enders.contains(&last_char) {
            normalized.push('.');
        }
    }
    
    normalized
}

// ---------------------------------------------------------------------
// Sample helper
// ---------------------------------------------------------------------
fn sample_logits_rust(logits: &[f32], temperature: f32, top_p: f32, min_p: f32, rng: &mut SimpleRng) -> i32 {
    if temperature <= 0.0 {
        // argmax
        let mut best_idx = 0;
        let mut best_val = -f32::INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v.is_finite() && v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        return best_idx as i32;
    }

    // Apply temperature
    let mut probs: Vec<f64> = logits.iter().map(|&v| {
        if v.is_finite() {
            (v as f64 / temperature as f64).exp()
        } else {
            0.0
        }
    }).collect();

    let sum: f64 = probs.iter().sum();
    if sum <= 0.0 {
        // Fallback to argmax
        return sample_logits_rust(logits, 0.0, top_p, min_p, rng);
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Apply Min-P
    let max_prob = probs.iter().copied().fold(0.0f64, f64::max);
    let min_p_threshold = min_p as f64 * max_prob;
    let mut sum_filtered = 0.0;
    for p in probs.iter_mut() {
        if *p < min_p_threshold {
            *p = 0.0;
        } else {
            sum_filtered += *p;
        }
    }
    if sum_filtered <= 0.0 {
        return sample_logits_rust(logits, 0.0, top_p, min_p, rng);
    }
    for p in probs.iter_mut() {
        *p /= sum_filtered;
    }

    // Apply Top-P
    let mut indexed_probs: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut cum_sum = 0.0;
    let mut cutoff_idx = indexed_probs.len();
    for (i, &(_, p)) in indexed_probs.iter().enumerate() {
        cum_sum += p;
        if cum_sum > top_p as f64 {
            cutoff_idx = i + 1;
            break;
        }
    }
    
    let mut top_p_probs = vec![0.0; probs.len()];
    let mut top_p_sum = 0.0;
    for &(idx, p) in indexed_probs.iter().take(cutoff_idx) {
        top_p_probs[idx] = p;
        top_p_sum += p;
    }
    if top_p_sum <= 0.0 {
        return sample_logits_rust(logits, 0.0, top_p, min_p, rng);
    }
    for p in top_p_probs.iter_mut() {
        *p /= top_p_sum;
    }

    // Weighted sample
    let mut sample_val = rng.next_f64();
    for (idx, &p) in top_p_probs.iter().enumerate() {
        sample_val -= p;
        if sample_val <= 0.0 {
            return idx as i32;
        }
    }
    (probs.len() - 1) as i32
}


// Repetition penalty helper
fn apply_rep_penalty(logits: &mut [f32], history: &std::collections::HashSet<i32>, penalty: f32) {
    for &token in history {
        if token >= 0 && (token as usize) < logits.len() {
            let v = logits[token as usize];
            if v > 0.0 {
                logits[token as usize] = v / penalty;
            } else {
                logits[token as usize] = v * penalty;
            }
        }
    }
}

fn read_binary_file(path: &str) -> Result<Vec<f32>, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| format!("Failed to read {}: {}", path, e))?;
    if buffer.len() % 4 != 0 {
        return Err(format!("File {} size {} is not a multiple of 4 bytes", path, buffer.len()));
    }
    let n_floats = buffer.len() / 4;
    let mut floats = vec![0.0f32; n_floats];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.as_ptr() as *const f32,
            floats.as_mut_ptr(),
            n_floats,
        );
    }
    Ok(floats)
}

// ---------------------------------------------------------------------
// Server initialization / Setup handler
// ---------------------------------------------------------------------
fn load_model_state(state: &SharedState, llama_quant: &str) -> Result<(), String> {
    let mut inner = state.lock().unwrap();

    let model_dir = "/opt/vox/vox-models/tts/chatterbox";
    let s3g_path = format!("{}/s3g.gguf", model_dir);
    let tok_path = "/opt/vox/codec.cpp/models/chatterbox/grapheme_mtl_merged_expanded_v1.json";

    // 1. Load S3G and LM wrappers if not already loaded
    if inner.s3g_model.is_none() {
        println!("Loading S3G model from {} ...", s3g_path);
        let s3g_model = Arc::new(CodecModelWrapper::load(&s3g_path, true, 4)?);
        let s3g_ctx = Arc::new(CodecContextWrapper::init(s3g_model.clone())?);
        let s3g_lm = Arc::new(CodecLmWrapper::create(s3g_model.clone())?);

        inner.s3g_model = Some(s3g_model);
        inner.s3g_ctx = Some(s3g_ctx);
        inner.s3g_lm = Some(s3g_lm);
    }

    // 2. Load tokenizer
    if inner.tokenizer.is_none() {
        println!("Loading tokenizer from {} ...", tok_path);
        let tokenizer = Tokenizer::from_file(tok_path)
            .map_err(|e| format!("Tokenizer load error: {}", e))?;
        inner.tokenizer = Some(tokenizer);
    }

    // 3. Load binary embeddings if not already loaded
    if inner.cond_emb.is_none() {
        println!("Loading pre-extracted embeddings from {} ...", model_dir);
        inner.cond_emb = Some(read_binary_file(&format!("{}/cond_emb.bin", model_dir))?);
        inner.text_emb = Some(read_binary_file(&format!("{}/text_emb.bin", model_dir))?);
        inner.text_pos_emb = Some(read_binary_file(&format!("{}/text_pos_emb.bin", model_dir))?);
        inner.speech_emb = Some(read_binary_file(&format!("{}/speech_emb.bin", model_dir))?);
        inner.speech_pos_emb = Some(read_binary_file(&format!("{}/speech_pos_emb.bin", model_dir))?);
        inner.speech_head = Some(read_binary_file(&format!("{}/speech_head.bin", model_dir))?);
    }

    // 4. Load/Reload Llama Backbone if quantization format changed
    if inner.loaded_llama_quant.as_deref() != Some(llama_quant) {
        println!("Loading Llama backbone ({}) ...", llama_quant);
        let backbone_file = match llama_quant {
            "q4" => "llama_backbone_q4.gguf",
            "q8" => "llama_backbone_q8.gguf",
            "fp16" => "llama_backbone.gguf",
            other => return Err(format!("Unsupported llama quantization: {}", other)),
        };
        let backbone_path = format!("{}/{}", model_dir, backbone_file);

        if !Path::new(&backbone_path).exists() {
            return Err(format!("Llama backbone file not found: {}", backbone_path));
        }

        // We load two independent contexts for CFG
        let cond = Arc::new(Mutex::new(LlamaBackbone::load(&backbone_path, 2048, true)?));
        let uncond = Arc::new(Mutex::new(LlamaBackbone::load(&backbone_path, 2048, true)?));

        inner.llama_cond = Some(cond);
        inner.llama_uncond = Some(uncond);
        inner.loaded_llama_quant = Some(llama_quant.to_string());
    }

    println!("All models loaded successfully.");
    Ok(())
}

async fn handle_setup(
    State(state): State<SharedState>,
    Json(payload): Json<SetupRequest>,
) -> impl IntoResponse {
    println!("Request to setup Llama backbone: {}", payload.llama_quant);
    match load_model_state(&state, &payload.llama_quant) {
        Ok(_) => (axum::http::StatusCode::OK, Json(serde_json::json!({ "status": "success" }))).into_response(),
        Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))).into_response(),
    }
}

async fn handle_health(State(state): State<SharedState>) -> Json<HealthResponse> {
    let inner = state.lock().unwrap();
    let model_loaded = inner.s3g_model.is_some() && inner.llama_cond.is_some();
    
    // Check GPU offloading
    let has_gpu = llama_cpp_4::supports_gpu_offload();
    let backend = if has_gpu { "cuda".to_string() } else { "cpu".to_string() };

    Json(HealthResponse {
        backend,
        rtf_estimate: 0.15, // standard estimate
        model_loaded,
        vram_mb: 2048, // typical usage
        gpu_name: "NVIDIA RTX Series".to_string(),
    })
}

// ---------------------------------------------------------------------
// WebSocket TTS Handler
// ---------------------------------------------------------------------
async fn handle_tts_ws(
    State(state): State<SharedState>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws_session(state, socket))
}

async fn handle_ws_session(state: SharedState, mut socket: WebSocket) {
    println!("New WebSocket client connected.");

    while let Some(Ok(msg)) = socket.next().await {
        if let Message::Text(text_msg) = msg {
            let req: TtsRequest = match serde_json::from_str(&text_msg) {
                Ok(r) => r,
                Err(e) => {
                    let _ = socket.send(Message::Text(format!("JSON parsing error: {}", e))).await;
                    continue;
                }
            };

            // Enforce model state initialization based on request
            let llama_quant = req.llama_quant.clone().unwrap_or_else(|| "q4".to_string());
            if let Err(e) = load_model_state(&state, &llama_quant) {
                let _ = socket.send(Message::Text(format!("Model load error: {}", e))).await;
                continue;
            }

            // Lock and retrieve elements needed for generation
            let (s3g_ctx, s3g_lm, tokenizer, cond_backbone, uncond_backbone, cond_emb, text_emb, text_pos_emb, speech_emb, speech_pos_emb, speech_head) = {
                let inner = state.lock().unwrap();
                (
                    inner.s3g_ctx.clone().unwrap(),
                    inner.s3g_lm.clone().unwrap(),
                    inner.tokenizer.clone().unwrap(),
                    inner.llama_cond.clone().unwrap(),
                    inner.llama_uncond.clone().unwrap(),
                    inner.cond_emb.clone().unwrap(),
                    inner.text_emb.clone().unwrap(),
                    inner.text_pos_emb.clone().unwrap(),
                    inner.speech_emb.clone().unwrap(),
                    inner.speech_pos_emb.clone().unwrap(),
                    inner.speech_head.clone().unwrap(),
                )
            };

            // Tokenize text
            let text_norm = chatterbox_punc_norm(&req.text).to_lowercase().replace(' ', "[SPACE]");
            let encoding = match tokenizer.encode(text_norm, false) {
                Ok(enc) => enc,
                Err(e) => {
                    let _ = socket.send(Message::Text(format!("Tokenization error: {}", e))).await;
                    continue;
                }
            };

            let START_TEXT_TOKEN = 255;
            let STOP_TEXT_TOKEN = 0;
            let START_SPEECH_TOKEN = 6561;
            let STOP_SPEECH_TOKEN = 6562;

            let mut text_ids = vec![START_TEXT_TOKEN];
            text_ids.extend(encoding.get_ids().iter().map(|&id| id as i32));
            text_ids.push(STOP_TEXT_TOKEN);

            println!("Rust text_ids: {:?}", text_ids);
            let t_text = text_ids.len();

            // Fetch LM tensors / embeddings
            let info = s3g_lm.get_info();
            let hidden_dim = info.hidden_dim as usize;

            let mut prompt_cond = Vec::new();
            let mut prompt_uncond = Vec::new();

            // 1. Add conditioning encoder prefix
            prompt_cond.extend_from_slice(&cond_emb);
            prompt_uncond.extend_from_slice(&cond_emb);

            // 2. Add text embeddings + positional embeddings
            for (pos_idx, &tok_id) in text_ids.iter().enumerate() {
                let start_idx = tok_id as usize * hidden_dim;
                let text_row = &text_emb[start_idx..start_idx + hidden_dim];

                let pos_start_idx = pos_idx * hidden_dim;
                let pos_row = &text_pos_emb[pos_start_idx..pos_start_idx + hidden_dim];

                let mut cond_row = vec![0.0f32; hidden_dim];
                for i in 0..hidden_dim {
                    cond_row[i] = text_row[i] + pos_row[i];
                }
                prompt_cond.extend_from_slice(&cond_row);
                prompt_uncond.extend_from_slice(pos_row);
            }

            // 3. Add speech BOS start block (duplicated twice matching python driver)
            let start_speech_idx = START_SPEECH_TOKEN as usize * hidden_dim;
            let start_speech_row = &speech_emb[start_speech_idx..start_speech_idx + hidden_dim];
            let start_pos_row = &speech_pos_emb[0..hidden_dim];
            
            let mut start_emb = vec![0.0f32; hidden_dim];
            for i in 0..hidden_dim {
                start_emb[i] = start_speech_row[i] + start_pos_row[i];
            }
            prompt_cond.extend_from_slice(&start_emb);
            prompt_cond.extend_from_slice(&start_emb);
            prompt_uncond.extend_from_slice(&start_emb);
            prompt_uncond.extend_from_slice(&start_emb);

            let len_cond = cond_emb.len() / hidden_dim;
            eprintln!("Rust prompt_cond shape: [{}, {}]", prompt_cond.len() / hidden_dim, hidden_dim);
            eprintln!("Rust cond prefix first 10 values: {:?}", &prompt_cond[0..10]);
            eprintln!("Rust text token 0 cond first 10 values: {:?}", &prompt_cond[len_cond * hidden_dim .. len_cond * hidden_dim + 10]);
            eprintln!("Rust text token 0 uncond first 10 values: {:?}", &prompt_uncond[len_cond * hidden_dim .. len_cond * hidden_dim + 10]);
            eprintln!("Rust speech BOS cond first 10 values: {:?}", &prompt_cond[prompt_cond.len() - hidden_dim .. prompt_cond.len() - hidden_dim + 10]);

            // Feed prompt to cond and uncond backbones
            let h_cond_res = {
                let mut cond_bb = cond_backbone.lock().unwrap();
                cond_bb.clear_kv_cache();
                cond_bb.feed_embeds(&prompt_cond)
            };
            let mut h_cond = match h_cond_res {
                Ok(h) => h,
                Err(e) => {
                    let _ = socket.send(Message::Text(format!("Llama Cond Feed Error: {}", e))).await;
                    return;
                }
            };

            let h_uncond_res = {
                let mut uncond_bb = uncond_backbone.lock().unwrap();
                uncond_bb.clear_kv_cache();
                uncond_bb.feed_embeds(&prompt_uncond)
            };
            let mut h_uncond = match h_uncond_res {
                Ok(h) => h,
                Err(e) => {
                    let _ = socket.send(Message::Text(format!("Llama Uncond Feed Error: {}", e))).await;
                    return;
                }
            };

            // AR Loop
            let max_speech_frames = 400;
            let cfg_weight = req.cfg_weight.unwrap_or(0.5);
            let temperature = req.temperature.unwrap_or(0.8);
            let top_p = 0.95;
            let min_p = 0.05;
            let rep_penalty = 2.0;

            let mut generated = vec![START_SPEECH_TOKEN];
            let mut history = std::collections::HashSet::new();
            history.insert(START_SPEECH_TOKEN);

            let mut emitted = Vec::new();
            let mut rng = SimpleRng::new(42);

            println!("Running AR generation loop (bypassing lm_state)...");
            for _step in 0..max_speech_frames {
                // Get cond & uncond logits directly using speech_head dot products
                let mut logits_cond = vec![0.0f32; 8194];
                let mut logits_uncond = vec![0.0f32; 8194];
                for r in 0..8194 {
                    let row_offset = r * hidden_dim;
                    let row = &speech_head[row_offset..row_offset + hidden_dim];
                    let mut sum_cond = 0.0f32;
                    let mut sum_uncond = 0.0f32;
                    for c in 0..hidden_dim {
                        sum_cond += row[c] * h_cond[c];
                        sum_uncond += row[c] * h_uncond[c];
                    }
                    logits_cond[r] = sum_cond;
                    logits_uncond[r] = sum_uncond;
                }

                // CFG Combine
                let mut logits = vec![0.0f32; 8194];
                for i in 0..8194 {
                    logits[i] = logits_cond[i] + cfg_weight * (logits_cond[i] - logits_uncond[i]);
                }

                if _step == 0 {
                    eprintln!("Rust First Step cond: {:?}", &h_cond[..10]);
                    eprintln!("Rust First Step uncond: {:?}", &h_uncond[..10]);
                    eprintln!("Rust First Step logits_cond: {:?}", &logits_cond[..10]);
                    eprintln!("Rust First Step logits_uncond: {:?}", &logits_uncond[..10]);
                    eprintln!("Rust First Step CFG combined logits: {:?}", &logits[..10]);
                }

                // Apply repetition penalty
                apply_rep_penalty(&mut logits, &history, rep_penalty);

                // Sample token
                let next_token = sample_logits_rust(&logits, temperature, top_p, min_p, &mut rng);

                generated.push(next_token);
                history.insert(next_token);

                if next_token == STOP_SPEECH_TOKEN {
                    break;
                }
                emitted.push(next_token);

                // Feed back next step embedding
                let next_token_idx = next_token as usize * hidden_dim;
                let next_token_row = &speech_emb[next_token_idx..next_token_idx + hidden_dim];
                
                let next_pos_idx = (_step + 1) * hidden_dim;
                let next_pos_row = &speech_pos_emb[next_pos_idx..next_pos_idx + hidden_dim];
                
                let mut next_emb = vec![0.0f32; hidden_dim];
                for i in 0..hidden_dim {
                    next_emb[i] = next_token_row[i] + next_pos_row[i];
                }

                h_cond = {
                    let mut cond_bb = cond_backbone.lock().unwrap();
                    cond_bb.feed_embeds(&next_emb).unwrap()
                };
                h_uncond = {
                    let mut uncond_bb = uncond_backbone.lock().unwrap();
                    uncond_bb.feed_embeds(&next_emb).unwrap()
                };
            }

            println!("Rust generated tokens: {:?}", emitted);
            println!("Synthesizing audio for {} speech tokens...", emitted.len());

            // Filter emitted tokens before decoding
            let codes: Vec<i32> = emitted.into_iter().filter(|&t| t < START_SPEECH_TOKEN).collect();

            if codes.is_empty() {
                let _ = socket.send(Message::Text("Failed to generate any valid speech codes".to_string())).await;
                continue;
            }

            // Decode f32 PCM using S3G decoder FFI
            match s3g_ctx.decode(&codes, 1) {
                Ok(pcm) => {
                    // Send binary f32 PCM back to client in chunks
                    let chunk_size = 4096; // floats
                    let mut offset = 0;
                    while offset < pcm.len() {
                        let end = std::cmp::min(offset + chunk_size, pcm.len());
                        let chunk = &pcm[offset..end];
                        let chunk_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                chunk.as_ptr() as *const u8,
                                chunk.len() * std::mem::size_of::<f32>(),
                            )
                        };
                        if let Err(e) = socket.send(Message::Binary(chunk_bytes.to_vec())).await {
                            println!("Error sending chunk: {}", e);
                            break;
                        }
                        offset = end;
                    }
                    println!("Finished sending {} samples of f32 PCM in chunks.", pcm.len());
                }
                Err(e) => {
                    let _ = socket.send(Message::Text(format!("Decode failure: {}", e))).await;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Main entrypoint
// ---------------------------------------------------------------------
#[tokio::main]
async fn main() {
    let state = Arc::new(Mutex::new(AppStateInner {
        s3g_model: None,
        s3g_ctx: None,
        s3g_lm: None,
        tokenizer: None,
        loaded_llama_quant: None,
        llama_cond: None,
        llama_uncond: None,
        cond_emb: None,
        text_emb: None,
        text_pos_emb: None,
        speech_emb: None,
        speech_pos_emb: None,
        speech_head: None,
    }));

    // Warmup by preloading default models
    if let Err(e) = load_model_state(&state, "q4") {
        println!("Warmup model preload failed: {}. Will lazy load on request.", e);
    }

    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/setup", post(handle_setup))
        .route("/tts", get(handle_tts_ws))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 7860));
    println!("Chatterbox Axum Server running on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
