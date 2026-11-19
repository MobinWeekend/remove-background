//! Sequential batch processing performed on the existing inference thread.
//!
//! Reusing the app's Engine avoids loading a second copy of a large ONNX model
//! and keeps GPU memory use predictable.

use std::collections::HashSet;
use std::fs;

use rembg_core::{Engine, ModelStore, imageops};

use super::{BatchEvent, BatchJob, ItemOutcome};

pub(crate) fn process(
    engine: &mut Engine,
    store: &ModelStore,
    job: BatchJob,
    mut emit: impl FnMut(BatchEvent),
) {
    let total = job.files.len();
    emit(BatchEvent::Started { id: job.id, total });

    if let Err(error) = fs::create_dir_all(&job.output_dir) {
        for index in 0..total {
            emit(BatchEvent::ItemStarted { id: job.id, index });
            emit(BatchEvent::ItemFinished {
                id: job.id,
                index,
                outcome: ItemOutcome::Failed { error: format!("Could not create output folder: {error}") },
            });
        }
        emit(BatchEvent::Finished { id: job.id, cancelled: false });
        return;
    }

    let mut destinations = HashSet::new();
    let mut completed = 0usize;
    for (index, input) in job.files.iter().enumerate() {
        if job.cancel.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }

        emit(BatchEvent::ItemStarted { id: job.id, index });
        let output = unique_output_path(input, &job.output_dir, &job.suffix, &mut destinations);
        let outcome = process_one(engine, store, &job, input, &output);
        emit(BatchEvent::ItemFinished { id: job.id, index, outcome });
        completed += 1;
    }

    // A cancel request received during the final image does not leave work
    // outstanding, so report the run as finished rather than cancelled.
    let cancelled = completed < total && job.cancel.load(std::sync::atomic::Ordering::Relaxed);
    emit(BatchEvent::Finished { id: job.id, cancelled });
}

fn process_one(
    engine: &mut Engine,
    store: &ModelStore,
    job: &BatchJob,
    input: &std::path::Path,
    output: &std::path::Path,
) -> ItemOutcome {
    let output = output.to_path_buf();

    // Never allow an empty suffix plus a same-folder PNG output to replace the
    // source image, even when the user enables overwrite.
    if let (Ok(source), Ok(target)) = (input.canonicalize(), output.canonicalize())
        && source == target
    {
        return ItemOutcome::Failed {
            error: "Output path points to the source image; choose a suffix or another folder".into(),
        };
    }

    if output.exists() && !job.overwrite {
        return ItemOutcome::Skipped { output };
    }

    let result = (|| -> Result<(), String> {
        let image = imageops::load(input).map_err(|e| format!("Could not read image: {e}"))?;
        let prediction = engine
            .predict(job.spec, &store.path(job.spec), &image)
            .map_err(|e| format!("Background removal failed: {e}"))?;
        let cutout = imageops::cutout(&image, &prediction.mask);
        cutout.save(&output).map_err(|e| format!("Could not save PNG: {e}"))?;
        Ok(())
    })();

    match result {
        Ok(()) => ItemOutcome::Saved { output },
        Err(error) => ItemOutcome::Failed { error },
    }
}

fn unique_output_path(
    input: &std::path::Path,
    output_dir: &std::path::Path,
    suffix: &str,
    used: &mut HashSet<std::path::PathBuf>,
) -> std::path::PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).filter(|s| !s.is_empty()).unwrap_or("image");
    let safe_suffix: String = suffix
        .chars()
        .filter(|c| !matches!(c, '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|'))
        .collect();
    let safe_suffix = safe_suffix.trim_end_matches(|c| c == ' ' || c == '.');

    let mut candidate = output_dir.join(format!("{stem}{safe_suffix}.png"));
    let mut index = 2usize;
    while !used.insert(candidate.clone()) {
        candidate = output_dir.join(format!("{stem}_{index}{safe_suffix}.png"));
        index += 1;
    }
    candidate
}
