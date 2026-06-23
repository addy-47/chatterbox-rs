//! Axum HTTP/WS server for chatterbox-rs.
//!
//! ## Routes
//!
//! - `POST /tts` — JSON `{ "text": "...", "language": "en" }` → WAV binary
//! - `GET /tts/stream` — WebSocket: text JSON in, binary PCM chunks out
//! - `GET /health` — JSON health check
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example tts_server
//! curl -X POST -H 'Content-Type: application/json' \
//!   -d '{"text":"Hello world"}' \
//!   http://localhost:7860/tts -o hello.wav
//! ```

use std::sync::Arc;
use std::sync::Mutex;
use std::pin::Pin;
use std::task::{Context, Poll};

use axum::{
    extract::{State, WebSocketUpgrade, ws::Message},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};

use serde::{Deserialize, Serialize};
use futures_util::stream::Stream;

use crate::{Engine, EngineError, EngineOptions};

// ── Shared state ──────────────────────────────────────────────────

struct AppState {
    engine: Mutex<Option<Engine>>,
    default_opts: Mutex<EngineOptions>,
}

// ─── Request / Response types ─────────────────────────────────────

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
    language: Option<String>,
}

#[derive(Deserialize)]
struct LoadRequest {
    t3_gguf_path: Option<String>,
    s3gen_gguf_path: Option<String>,
    language: Option<String>,
    gpu_layers: Option<i32>,
    cfm_steps: Option<i32>,
    stream_chunk_tokens: Option<i32>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    models_loaded: bool,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ─── Routes ───────────────────────────────────────────────────────

// ─── Routes ───────────────────────────────────────────────────────

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let engine_lock = state.engine.lock().unwrap();
    let loaded = engine_lock.is_some();
    Json(HealthResponse {
        status: "ok",
        models_loaded: loaded,
    })
}

async fn load_models_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let mut engine_lock = state.engine.lock().unwrap();
    
    // Drop existing engine to free VRAM
    *engine_lock = None;

    let mut opts = state.default_opts.lock().unwrap().clone();
    if let Some(p) = req.t3_gguf_path {
        opts.t3_gguf_path = p;
    }
    if let Some(p) = req.s3gen_gguf_path {
        opts.s3gen_gguf_path = p;
    }
    if let Some(l) = req.language {
        opts.language = l;
    }
    if let Some(g) = req.gpu_layers {
        opts.n_gpu_layers = g;
    }
    if let Some(c) = req.cfm_steps {
        opts.cfm_steps = c;
        opts.stream_cfm_steps = c;
    }
    if let Some(s) = req.stream_chunk_tokens {
        opts.stream_chunk_tokens = s;
    }

    match Engine::new(opts) {
        Ok(engine) => {
            *engine_lock = Some(engine);
            Ok(Json(serde_json::json!({ "status": "loaded" })))
        }
        Err(e) => {
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to load models: {:?}", e),
                }),
            ))
        }
    }
}

async fn unload_models_handler(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let mut engine_lock = state.engine.lock().unwrap();
    *engine_lock = None;
    Json(serde_json::json!({ "status": "unloaded" }))
}

async fn tts_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TtsRequest>,
) -> Result<Vec<u8>, (StatusCode, Json<ErrorResponse>)> {
    if req.text.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "text is required".into(),
            }),
        ));
    }

    let engine_lock = state.engine.lock().unwrap();
    let engine = match &*engine_lock {
        Some(e) => e,
        None => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "models not loaded; POST to /models/load first".into(),
                }),
            ));
        }
    };

    let result = engine
        .synthesize(&req.text)
        .map_err(|e| map_error(e))?;
    
    // Explicitly drop before sending bytes
    drop(engine_lock);

    Ok(encode_wav(&result.pcm, result.sample_rate as u32))
}

async fn tts_stream_pcm_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TtsRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    if req.text.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "text is required".into(),
            }),
        ));
    }

    {
        let engine_lock = state.engine.lock().unwrap();
        if engine_lock.is_none() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "models not loaded; POST to /models/load first".into(),
                }),
            ));
        }
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(16);

    // Spawn blocking task to run the engine.
    tokio::task::spawn_blocking(move || {
        let engine_lock = state.engine.lock().unwrap();
        let engine = match &*engine_lock {
            Some(e) => e,
            None => {
                let _ = tx.blocking_send(Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "models not loaded",
                )));
                return;
            }
        };

        let res = engine.synthesize_streaming(&req.text, |chunk, _chunk_idx, _is_last| {
            let bytes: Vec<u8> = chunk
                .iter()
                .flat_map(|&s| s.to_le_bytes())
                .collect();
            // Send chunk to receiver
            if tx.blocking_send(Ok(axum::body::Bytes::from(bytes))).is_err() {
                // Receiver was dropped, stop or ignore.
            }
        });

        if let Err(e) = res {
            let err_msg = format!("synthesis failed: {:?}", e);
            let _ = tx.blocking_send(Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                err_msg,
            )));
        }
    });

    struct ReceiverStream {
        inner: tokio::sync::mpsc::Receiver<Result<axum::body::Bytes, std::io::Error>>,
    }

    impl Stream for ReceiverStream {
        type Item = Result<axum::body::Bytes, std::io::Error>;

        fn poll_next(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            self.inner.poll_recv(cx)
        }
    }

    let body = axum::body::Body::from_stream(ReceiverStream { inner: rx });

    let response = Response::builder()
        .header("Content-Type", "audio/pcm-f32le; rate=24000")
        .body(body)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("failed to build response: {}", e),
                }),
            )
        })?;

    Ok(response)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |mut socket| async move {
        while let Some(msg) = socket.recv().await {
            let msg = match msg {
                Ok(m) => m,
                Err(_) => break,
            };

            let req: TtsRequest = match serde_json::from_slice(&msg.into_data()) {
                Ok(r) => r,
                Err(e) => {
                    let err = serde_json::to_string(&ErrorResponse {
                        error: format!("invalid JSON: {}", e),
                    })
                    .unwrap_or_default();
                    let _ = socket.send(Message::Text(err)).await;
                    continue;
                }
            };

            if req.text.trim().is_empty() {
                let err = serde_json::to_string(&ErrorResponse {
                    error: "text is required".into(),
                })
                .unwrap_or_default();
                let _ = socket.send(Message::Text(err)).await;
                continue;
            }

            // Lock engine, synthesize, release lock, then send.
            // This avoids holding a non-Send MutexGuard across .await.
            let pcm_result = {
                let engine_lock = state.engine.lock().unwrap();
                if let Some(engine) = &*engine_lock {
                    engine.synthesize(&req.text).map(|r| r.pcm)
                } else {
                    Err(EngineError::LoadFailed("Models not loaded".into()))
                }
            };

            let pcm = match pcm_result {
                Ok(p) => p,
                Err(e) => {
                    let msg = serde_json::to_string(&ErrorResponse {
                        error: e.to_string(),
                    })
                    .unwrap_or_default();
                    let _ = socket.send(Message::Text(msg)).await;
                    continue;
                }
            };

            // Send PCM in binary chunks.
            for chunk in pcm.chunks(4096) {
                let bytes: Vec<u8> = chunk
                    .iter()
                    .flat_map(|&s| s.to_le_bytes())
                    .collect();
                if socket.send(Message::Binary(bytes)).await.is_err() {
                    break;
                }
            }
        }
    })
}

// ── WAV encoding ──────────────────────────────────────────────────

/// Encode PCM f32 samples into a WAV file (16-bit mono).
fn encode_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_size = pcm.len() * 2; // 16-bit samples
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + data_size);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(file_size as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt sub-chunk (PCM, 1 channel, 16-bit)
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes());  // audio format (PCM)
    wav.extend_from_slice(&1u16.to_le_bytes());  // num channels
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes());  // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data sub-chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_size as u32).to_le_bytes());

    // PCM samples (f32 → i16)
    for &sample in pcm {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        wav.extend_from_slice(&i16_sample.to_le_bytes());
    }

    wav
}

// ── Error mapping ─────────────────────────────────────────────────

fn map_error(e: EngineError) -> (StatusCode, Json<ErrorResponse>) {
    match e {
        EngineError::LoadFailed(msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("engine load failed: {}", msg),
            }),
        ),
        EngineError::SynthesisFailed(msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("synthesis failed: {}", msg),
            }),
        ),
        EngineError::Cancelled => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "synthesis cancelled".into(),
            }),
        ),
    }
}

// ── Server builder ────────────────────────────────────────────────

/// Options for the HTTP/WS TTS server.
pub struct ServerOptions {
    /// Bind address.  Default: `0.0.0.0:7860`.
    pub bind: String,

    /// Engine options passed to [`Engine::new`].
    pub engine: EngineOptions,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:7860".into(),
            engine: EngineOptions::default(),
        }
    }
}

/// Start the Chatterbox TTS server.
///
/// Returns a [`tokio::task::JoinHandle`] for graceful shutdown.
pub async fn run(opts: ServerOptions) -> Result<(), Box<dyn std::error::Error>> {
    let state = Arc::new(AppState {
        engine: Mutex::new(None),
        default_opts: Mutex::new(opts.engine),
    });

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/models/load", post(load_models_handler))
        .route("/models/unload", post(unload_models_handler))
        .route("/tts", post(tts_handler))
        .route("/tts/stream", get(ws_handler))
        .route("/tts/stream-pcm", post(tts_stream_pcm_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&opts.bind).await?;
    println!("Chatterbox TTS server listening on {}", opts.bind);

    axum::serve(listener, app).await?;

    Ok(())
}
