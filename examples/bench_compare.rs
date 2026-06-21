/// Comparison benchmark: synthesize text with Rust bindings, compare against
/// CLI-generated reference WAV files.
///
/// Usage:
///   cargo run --example bench_compare --features server
///
/// This example:
///   1. Reads each CLI reference WAV from bench_audio/
///   2. Synthesizes the same text with the Rust Engine (seed=42)
///   3. Writes Rust output WAV to bench_audio/rust_<lang>.wav
///   4. Computes MSE and peak difference vs reference
///
/// Requires model files at /opt/vox-models/tts/chatterbox/.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

fn main() {
    let model_dir = PathBuf::from("/opt/vox-models/tts/chatterbox");
    let base_t3 = model_dir.join("t3-q4_0.gguf");
    let base_s3 = model_dir.join("s3gen-f16.gguf");

    if !base_t3.exists() || !base_s3.exists() {
        eprintln!("Model files not found at {}", model_dir.display());
        std::process::exit(1);
    }

    // Test cases: (lang, text, reference_wav)
    let cases: Vec<(&str, &str, &str)> = vec![
        (
            "en",
            "Good morning, this is a test of the Chatterbox text to speech system running entirely on CPU. The quick brown fox jumps over the lazy dog.",
            "en_good_morning.wav",
        ),
        (
            "de",
            "Guten Tag, dies ist ein Test des Chatterbox Text zu Sprache Systems, das vollständig auf der CPU läuft.",
            "de_guten_tag.wav",
        ),
        (
            "es",
            "Buenos días, esta es una prueba del sistema de síntesis de voz Chatterbox funcionando completamente en CPU.",
            "es_buenos_dias.wav",
        ),
        (
            "fr",
            "Bonjour, ceci est un test du système de synthèse vocale Chatterbox fonctionnant entièrement sur processeur.",
            "fr_bonjour.wav",
        ),
        (
            "it",
            "Buongiorno, questo è un test del sistema di sintesi vocale Chatterbox che funziona interamente sulla CPU.",
            "it_buongiorno.wav",
        ),
        (
            "ko",
            "안녕하세요, 이것은 CPU에서 완전히 실행되는 Chatterbox 음성 합성 시스템의 테스트입니다.",
            "ko_annyeong.wav",
        ),
    ];

    let bench_dir = PathBuf::from("bench_audio");
    if !bench_dir.exists() {
        eprintln!("bench_audio/ directory not found (run from crate root)");
        std::process::exit(1);
    }

    let mut all_pass = true;

    for &(lang, text, ref_wav_name) in &cases {
        let ref_path = bench_dir.join(ref_wav_name);
        if !ref_path.exists() {
            eprintln!("[{}] Reference WAV not found: {}", lang, ref_path.display());
            all_pass = false;
            continue;
        }

        eprint!("[{}] Loading engine and synthesizing... ", lang);

        // Create engine with seed=42 to match CLI benchmark
        let opts = chatterbox_rs::EngineOptions {
            t3_gguf_path: base_t3.to_string_lossy().into_owned(),
            s3gen_gguf_path: base_s3.to_string_lossy().into_owned(),
            language: lang.to_string(),
            seed: 42,
            n_gpu_layers: 99, // use GPU
            ..Default::default()
        };

        let engine = match chatterbox_rs::Engine::new(opts) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("FAILED to create engine: {}", err);
                all_pass = false;
                continue;
            }
        };

        let result = match engine.synthesize(text) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("FAILED to synthesize: {}", err);
                all_pass = false;
                continue;
            }
        };

        // Write Rust output WAV
        let rust_wav_name = format!("rust_{}.wav", lang);
        let rust_path = bench_dir.join(&rust_wav_name);
        write_wav(&rust_path, &result.pcm, result.sample_rate as u32)
            .expect("Failed to write Rust WAV");

        // Read reference WAV and compare
        let (ref_samples, ref_rate) = read_wav(&ref_path).expect("Failed to read reference WAV");

        if ref_rate != result.sample_rate {
            eprintln!(
                "WARN: sample rate mismatch: ref={} rust={}",
                ref_rate, result.sample_rate
            );
        }

        // Compute comparison metrics
        let min_len = ref_samples.len().min(result.pcm.len());
        if min_len == 0 {
            eprintln!("[{}] One of the audio files is empty", lang);
            all_pass = false;
            continue;
        }

        let mut mse = 0.0f64;
        let mut peak_diff = 0.0f64;
        for i in 0..min_len {
            let d = (ref_samples[i] - result.pcm[i]) as f64;
            mse += d * d;
            let abs_d = d.abs();
            if abs_d > peak_diff {
                peak_diff = abs_d;
            }
        }
        mse /= min_len as f64;
        let rmse = mse.sqrt();

        let max_amp = ref_samples
            .iter()
            .chain(result.pcm.iter())
            .map(|s| s.abs() as f64)
            .fold(0.0f64, f64::max);

        let snr_db = if mse > 1e-12 {
            10.0 * (max_amp * max_amp / mse).log10()
        } else {
            999.0 // essentially identical
        };

        eprintln!(
            "OK  |  samples: ref={} rust={} (min={})  |  RMSE={:.6}  peak_diff={:.4}  SNR={:.1} dB",
            ref_samples.len(),
            result.pcm.len(),
            min_len,
            rmse,
            peak_diff,
            snr_db,
        );

        if peak_diff > 2.0 {
            eprintln!(
                "  ⚠  Large peak difference ({:.4}) — GPU vs CPU numerical differences expected",
                peak_diff
            );
        }
    }

    if all_pass {
        eprintln!("\n✅ All benchmarks completed successfully.");
    } else {
        eprintln!("\n❌ Some benchmarks failed.");
        std::process::exit(1);
    }
}

// ─── WAV I/O helpers ──────────────────────────────────────────────

/// Write f32 PCM samples as a 16-bit mono WAV file.
fn write_wav(path: &Path, pcm: &[f32], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let data_size = pcm.len() * 2;
    let file_size = 36 + data_size;

    use std::io::Write;
    let mut f = File::create(path)?;

    f.write_all(b"RIFF")?;
    f.write_all(&(file_size as u32).to_le_bytes())?;
    f.write_all(b"WAVE")?;

    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;

    f.write_all(b"data")?;
    f.write_all(&(data_size as u32).to_le_bytes())?;

    for &sample in pcm {
        let clamped = sample.clamp(-1.0, 1.0);
        f.write_all(&((clamped * 32767.0) as i16).to_le_bytes())?;
    }

    Ok(())
}

/// Read a 16-bit mono WAV file and return (PCM f32 samples, sample_rate).
fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    if buf.len() < 44 {
        return Err("File too small to be a WAV".into());
    }
    if &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }

    // fmt chunk
    let fmt_size = u32::from_le_bytes(buf[16..20].try_into()?) as usize;
    let audio_fmt = u16::from_le_bytes(buf[20..22].try_into()?);
    let num_channels = u16::from_le_bytes(buf[22..24].try_into()?);
    let sample_rate = u32::from_le_bytes(buf[24..28].try_into()?);
    let bits_per_sample = u16::from_le_bytes(buf[34..36].try_into()?);

    if audio_fmt != 1 {
        return Err("Only PCM format supported".into());
    }
    if num_channels != 1 {
        return Err("Only mono supported".into());
    }
    if bits_per_sample != 16 {
        return Err("Only 16-bit supported".into());
    }

    // Find data chunk
    let mut offset = 12 + 8 + fmt_size; // skip first fmt chunk
    loop {
        if offset + 8 > buf.len() {
            return Err("No data chunk found".into());
        }
        let chunk_id = &buf[offset..offset + 4];
        let chunk_size = u32::from_le_bytes(buf[offset + 4..offset + 8].try_into()?) as usize;
        if chunk_id == b"data" {
            let data = &buf[offset + 8..offset + 8 + chunk_size.min(buf.len() - offset - 8)];
            let num_samples = data.len() / 2;
            let mut pcm = Vec::with_capacity(num_samples);
            for i in 0..num_samples {
                let sample = i16::from_le_bytes(data[i * 2..i * 2 + 2].try_into()?);
                pcm.push(sample as f32 / 32767.0);
            }
            return Ok((pcm, sample_rate));
        }
        offset += 8 + chunk_size;
        if offset >= buf.len() {
            return Err("No data chunk found".into());
        }
    }
}
