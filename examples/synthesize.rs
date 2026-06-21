/// Simple CLI example: synthesise text and write a WAV.
///
/// Usage:
///   cargo run --example synthesize -- \
///       --text "Hello, world." \
///       --out hello.wav \
///       --language en
///
/// Model paths default to /opt/vox-models/tts/chatterbox/.
/// Override with --t3-gguf and --s3gen-gguf.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut text = String::new();
    let mut out_path = PathBuf::from("output.wav");
    let mut t3_path = PathBuf::from("/opt/vox-models/tts/chatterbox/t3-q4_0.gguf");
    let mut s3_path = PathBuf::from("/opt/vox-models/tts/chatterbox/s3gen-f16.gguf");
    let mut language = "en".to_string();
    let mut n_gpu_layers = 0;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--text" => { i += 1; text = args[i].clone(); }
            "--out" => { i += 1; out_path = args[i].clone().into(); }
            "--t3-gguf" => { i += 1; t3_path = args[i].clone().into(); }
            "--s3gen-gguf" => { i += 1; s3_path = args[i].clone().into(); }
            "--language" => { i += 1; language = args[i].clone(); }
            "--gpu-layers" => { i += 1; n_gpu_layers = args[i].parse()?; }
            "--verbose" => { verbose = true; }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if text.is_empty() {
        eprintln!("Usage: synthesize --text \"...\" [--out path] [--language en]");
        std::process::exit(1);
    }

    let opts = chatterbox_rs::EngineOptions {
        t3_gguf_path: t3_path.to_string_lossy().into_owned(),
        s3gen_gguf_path: s3_path.to_string_lossy().into_owned(),
        language,
        n_gpu_layers,
        verbose,
        ..Default::default()
    };

    let engine = chatterbox_rs::Engine::new(opts)?;
    let result = engine.synthesize(&text)?;

    // Write WAV manually (no hound dependency).
    let data_size = result.pcm.len() * 2;
    let file_size = 36 + data_size;
    let sample_rate = result.sample_rate;

    use std::io::Write;
    let mut f = std::fs::File::create(&out_path)?;

    f.write_all(b"RIFF")?;
    f.write_all(&(file_size as u32).to_le_bytes())?;
    f.write_all(b"WAVE")?;

    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;  // chunk size
    f.write_all(&1u16.to_le_bytes())?;   // PCM
    f.write_all(&1u16.to_le_bytes())?;   // mono
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?;   // block align
    f.write_all(&16u16.to_le_bytes())?;  // bits per sample

    f.write_all(b"data")?;
    f.write_all(&(data_size as u32).to_le_bytes())?;

    for &sample in &result.pcm {
        let clamped = sample.clamp(-1.0, 1.0);
        f.write_all(&((clamped * 32767.0) as i16).to_le_bytes())?;
    }

    eprintln!(
        "Wrote {} ({} samples, {:.1}s, T3: {:.0}ms, S3Gen: {:.0}ms)",
        out_path.display(),
        result.audio_samples,
        result.audio_samples as f64 / result.sample_rate as f64,
        result.t3_ms,
        result.s3gen_ms,
    );

    Ok(())
}
