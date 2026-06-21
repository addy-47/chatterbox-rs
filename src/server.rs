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

use axum::{
    extract::{State, WebSocketUpgrade, ws::Message},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};

use serde::{Deserialize, Serialize};

use crate::{Engine, EngineError, EngineOptions};

// ── Shared state ──────────────────────────────────────────────────

struct AppState {
    engine: Mutex<Engine>,
}

// ─── Request / Response types ─────────────────────────────────────

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
    language: Option<String>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ─── Routes ───────────────────────────────────────────────────────

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    // Verify the mutex is not poisoned.
    drop(state.engine.lock().unwrap());
    Json(HealthResponse { status: "ok" })
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

    let engine = state.engine.lock().unwrap();
    let result = engine
        .synthesize(&req.text)
        .map_err(|e| map_error(e))?;
    // Drop the lock before async work.
    drop(engine);

    Ok(encode_wav(&result.pcm, result.sample_rate as u32))
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
                let engine = state.engine.lock().unwrap();
                engine.synthesize(&req.text).map(|r| r.pcm)
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
    let engine = Engine::new(opts.engine)?;

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
    });

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/tts", post(tts_handler))
        .route("/tts/stream", get(ws_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&opts.bind).await?;
    println!("Chatterbox TTS server listening on {}", opts.bind);

    axum::serve(listener, app).await?;

    Ok(())
}
