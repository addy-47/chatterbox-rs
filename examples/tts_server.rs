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
    let args = std::env::args().collect::<Vec<String>>();
    
    let mut t3_gguf = "/home/hypr4/.vox/models/tts/chatterbox/t3-q4_0.gguf".to_string();
    let mut s3gen_gguf = "/home/hypr4/.vox/models/tts/chatterbox/s3gen-f16.gguf".to_string();
    let mut port = 7860u16;
    let mut gpu_layers = 99i32;
    let mut cfm_steps = 10i32;
    let mut language = "en".to_string();
    let mut bind_addr = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--t3-gguf" => {
                if i + 1 < args.len() {
                    t3_gguf = args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing value for --t3-gguf".into());
                }
            }
            "--s3gen-gguf" => {
                if i + 1 < args.len() {
                    s3gen_gguf = args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing value for --s3gen-gguf".into());
                }
            }
            "--port" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --port".into());
                }
            }
            "--gpu-layers" => {
                if i + 1 < args.len() {
                    gpu_layers = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --gpu-layers".into());
                }
            }
            "--cfm-steps" => {
                if i + 1 < args.len() {
                    cfm_steps = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --cfm-steps".into());
                }
            }
            "--language" => {
                if i + 1 < args.len() {
                    language = args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing value for --language".into());
                }
            }
            "--bind" => {
                if i + 1 < args.len() {
                    bind_addr = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing value for --bind".into());
                }
            }
            h if h == "--help" || h == "-h" => {
                println!("Usage: tts_server [OPTIONS]");
                println!("Options:");
                println!("  --t3-gguf <PATH>       Path to T3 GGUF model");
                println!("  --s3gen-gguf <PATH>    Path to S3Gen GGUF model");
                println!("  --port <PORT>          Port to listen on (default: 7860)");
                println!("  --bind <ADDR>          Bind address (e.g. 0.0.0.0:7860, overrides --port)");
                println!("  --gpu-layers <NUM>     Number of GPU layers (default: 99)");
                println!("  --cfm-steps <NUM>      CFM steps (default: 10)");
                println!("  --language <LANG>      Default language (default: en)");
                return Ok(());
            }
            _ => {
                return Err(format!("Unknown argument: {}", args[i]).into());
            }
        }
    }

    let bind = bind_addr.unwrap_or_else(|| format!("0.0.0.0:{}", port));

    let opts = chatterbox_rs::server::ServerOptions {
        bind,
        engine: chatterbox_rs::EngineOptions {
            t3_gguf_path: t3_gguf,
            s3gen_gguf_path: s3gen_gguf,
            language,
            n_gpu_layers: gpu_layers,
            cfm_steps,
            stream_cfm_steps: cfm_steps,
            ..Default::default()
        },
    };

    chatterbox_rs::server::run(opts).await
}
