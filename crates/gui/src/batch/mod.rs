//! Batch background removal. Kept separate from the main viewer UI so this
//! feature can be developed or ported independently.

pub(crate) mod processor;
mod window;

pub use window::BatchWindow;

use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

use rembg_core::ModelSpec;

pub(crate) struct BatchJob {
    pub id: u64,
    pub files: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub suffix: String,
    pub skip_existing: bool,
    pub spec: &'static ModelSpec,
    pub cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub(crate) enum ItemOutcome {
    Saved { output: PathBuf },
    Skipped { output: PathBuf },
    Failed { error: String },
}

#[derive(Debug)]
pub(crate) enum BatchEvent {
    Started { id: u64, total: usize },
    ItemStarted { id: u64, index: usize },
    ItemFinished { id: u64, index: usize, outcome: ItemOutcome },
    Finished { id: u64, cancelled: bool },
}
