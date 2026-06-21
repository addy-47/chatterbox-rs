use std::fmt;

/// Errors returned by the Chatterbox TTS engine.
#[derive(Debug)]
pub enum EngineError {
    /// The engine could not be created (model loading failed, etc.).
    LoadFailed(String),

    /// Synthesis failed (text too long, model error, etc.).
    SynthesisFailed(String),

    /// Synthesis was cancelled by a concurrent call to [`crate::Engine::cancel`].
    Cancelled,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::LoadFailed(msg) => write!(f, "Engine load failed: {}", msg),
            EngineError::SynthesisFailed(msg) => write!(f, "Synthesis failed: {}", msg),
            EngineError::Cancelled => write!(f, "Synthesis cancelled"),
        }
    }
}

impl std::error::Error for EngineError {}
