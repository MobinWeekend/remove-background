//! Headless background-removal engine.
//!
//! ```no_run
//! use rembg_core::{Engine, DevicePref, ModelStore, imageops};
//! let store = ModelStore::open_default()?;
//! let spec = store.default_model();
//! let mut engine = Engine::new(DevicePref::Auto)?;
//! let img = imageops::load("photo.jpg")?;
//! let pred = engine.predict(spec, &store.path(spec), &img)?;
//! imageops::cutout(&img, &pred.mask).save("photo.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod engine;
pub mod grid;
pub mod imageops;
pub mod registry;
pub mod store;

pub use engine::{ActiveDevice, DevicePref, Engine, Prediction};
pub use registry::{Family, MODELS, ModelSpec};
pub use store::ModelStore;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("onnxruntime: {0}")]
    Ort(String),
    #[error("download failed: {0}")]
    Download(String),
    #[error("checksum mismatch for {model}: expected {expected}, got {got}")]
    Checksum { model: String, expected: String, got: String },
    #[error("cancelled")]
    Cancelled,
    #[error("unknown model '{0}'")]
    UnknownModel(String),
    #[error("model '{0}' is not installed (run: rembg models pull {0})")]
    NotInstalled(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
