/// Axum HTTP/WS TTS server example.
///
/// Usage:
///   cargo run --example tts_server --features server
///
/// Then:
///   curl -X POST -H 'Content-Type: application/json' \
///     -d '{"text":"Hello world","language":"en"}' \
///     http://localhost:7860/tts -o hello.wav

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = chatterbox_rs::server::ServerOptions {
        bind: "0.0.0.0:7860".into(),
        engine: chatterbox_rs::EngineOptions {
            t3_gguf_path: "/opt/vox-models/tts/chatterbox/t3-q4_0.gguf".into(),
            s3gen_gguf_path: "/opt/vox-models/tts/chatterbox/s3gen-f16.gguf".into(),
            language: "en".into(),
            n_gpu_layers: 99,
            ..Default::default()
        },
    };

    chatterbox_rs::server::run(opts).await
}
