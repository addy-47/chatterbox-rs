use std::sync::Arc;
use chatterbox_rs::{CodecModelWrapper, CodecLmWrapper};

fn main() {
    println!("Loading codec model...");
    let model_path = "/opt/vox/vox-models/tts/chatterbox/s3g.gguf";
    match CodecModelWrapper::load(model_path, false, 4) {
        Ok(model) => {
            println!("Model loaded successfully!");
            println!("Arch: {:?}", model.arch());
            println!("Name: {}", model.name());
            
            let model_arc = Arc::new(model);
            println!("Attempting to create codec_lm...");
            match CodecLmWrapper::create(model_arc) {
                Ok(_lm) => {
                    println!("SUCCESS: codec_lm created successfully!");
                }
                Err(e) => {
                    println!("ERROR: codec_lm creation failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("ERROR: failed to load model: {}", e);
        }
    }
}
