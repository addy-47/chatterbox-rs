use chatterbox_rs::{Engine, EngineOptions};

/// These tests require the model files at the standard paths.
/// Skip if the files are not present.
fn model_paths() -> Option<(String, String)> {
    let t3 = std::path::Path::new("/opt/vox-models/tts/chatterbox/t3-q4_0.gguf");
    let s3 = std::path::Path::new("/opt/vox-models/tts/chatterbox/s3gen-f16.gguf");
    if t3.exists() && s3.exists() {
        Some((t3.to_string_lossy().into(), s3.to_string_lossy().into()))
    } else {
        None
    }
}

#[test]
fn test_engine_create_default_options() {
    let (t3, s3) = match model_paths() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model files not found");
            return;
        }
    };

    let engine = Engine::new(EngineOptions {
        t3_gguf_path: t3,
        s3gen_gguf_path: s3,
        language: "en".into(),
        ..Default::default()
    });

    assert!(engine.is_ok(), "Engine should load: {:?}", engine.err());
}

#[test]
fn test_synthesize_english() {
    let (t3, s3) = match model_paths() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model files not found");
            return;
        }
    };

    let engine = Engine::new(EngineOptions {
        t3_gguf_path: t3,
        s3gen_gguf_path: s3,
        language: "en".into(),
        ..Default::default()
    })
    .expect("Engine should load");

    let result = engine.synthesize("Hello, world.").expect("Synthesis should succeed");
    assert!(result.pcm.len() > 0, "PCM should not be empty");
    assert_eq!(result.sample_rate, 24000, "Sample rate should be 24000");
    assert!(result.t3_tokens > 0, "Should have generated speech tokens");
    assert!(result.audio_samples > 0, "Should have audio samples");
}

#[test]
fn test_reject_empty_text() {
    let (t3, s3) = match model_paths() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model files not found");
            return;
        }
    };

    let engine = Engine::new(EngineOptions {
        t3_gguf_path: t3,
        s3gen_gguf_path: s3,
        language: "en".into(),
        ..Default::default()
    })
    .expect("Engine should load");

    let result = engine.synthesize("");
    assert!(result.is_err(), "Empty text should error");
}

#[test]
fn test_engine_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Engine>();
}

#[test]
fn test_multiple_calls_reuse_engine() {
    let (t3, s3) = match model_paths() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model files not found");
            return;
        }
    };

    let engine = Engine::new(EngineOptions {
        t3_gguf_path: t3,
        s3gen_gguf_path: s3,
        language: "en".into(),
        top_k: 50,   // faster sampling for test
        n_predict: 200, // shorter output for test
        ..Default::default()
    })
    .expect("Engine should load");

    for _ in 0..3 {
        let result = engine.synthesize("Test.").expect("Synthesis should succeed");
        assert!(result.pcm.len() > 0, "PCM should not be empty");
    }
}

#[test]
fn test_different_languages() {
    let (t3, s3) = match model_paths() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: model files not found");
            return;
        }
    };

    // Test a few languages with short text.
    for lang in &["en", "es", "fr", "de"] {
        let engine = Engine::new(EngineOptions {
            t3_gguf_path: t3.clone(),
            s3gen_gguf_path: s3.clone(),
            language: lang.to_string(),
            n_predict: 200,
            ..Default::default()
        });

        match engine {
            Ok(e) => {
                let text = match *lang {
                    "en" => "Hello.",
                    "es" => "Hola.",
                    "fr" => "Bonjour.",
                    "de" => "Hallo.",
                    _ => "Hello.",
                };
                let result = e.synthesize(text);
                assert!(result.is_ok(), "Synthesis in {} should succeed: {:?}", lang, result.err());
                if let Ok(r) = result {
                    assert!(r.pcm.len() > 0, "PCM for {} should not be empty", lang);
                }
            }
            Err(e) => {
                eprintln!("Engine load for {} failed (may be expected): {}", lang, e);
            }
        }
    }
}

#[test]
fn test_error_on_missing_model() {
    let result = Engine::new(EngineOptions {
        t3_gguf_path: "/nonexistent/t3.gguf".into(),
        s3gen_gguf_path: "/nonexistent/s3gen.gguf".into(),
        ..Default::default()
    });
    assert!(result.is_err(), "Missing model should error");
    let msg = result.err().unwrap().to_string();
    assert!(msg.contains("not found"), "Error should mention 'not found': {}", msg);
}
