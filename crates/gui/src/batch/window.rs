//! Secondary viewport and controls for batch background removal.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::Ordering};

use egui::{Color32, RichText};
use rembg_core::{ModelStore, registry};

use crate::worker::Worker;

use super::{BatchEvent, BatchJob, ItemOutcome};

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"];

struct QueueItem {
    path: PathBuf,
    state: ItemState,
}

#[derive(Default)]
enum ItemState {
    #[default]
    Queued,
    Processing,
    Saved,
    Skipped,
    Failed(String),
}

struct ActiveBatch {
    id: u64,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    total: usize,
    completed: usize,
}

pub struct BatchWindow {
    pub open: bool,
    items: Vec<QueueItem>,
    output_dir: PathBuf,
    output_dir_set: bool,
    recursive: bool,
    suffix: String,
    overwrite: bool,
    selected_model: String,
    next_job_id: u64,
    active: Option<ActiveBatch>,
    waiting_for_model: Option<&'static str>,
    notice: Option<String>,
}

impl BatchWindow {
    pub fn new(default_model: &str) -> Self {
        Self {
            open: false,
            items: Vec::new(),
            output_dir: std::env::current_dir()
                .unwrap_or_default()
                .join("Background Removed"),
            output_dir_set: false,
            recursive: false,
            suffix: "_nobg".into(),
            overwrite: false,
            selected_model: default_model.into(),
            next_job_id: 1,
            active: None,
            waiting_for_model: None,
            notice: None,
        }
    }

    pub fn handle_event(&mut self, event: BatchEvent) {
        match event {
            BatchEvent::Started { id, total } => {
                if let Some(active) = self.active.as_mut().filter(|a| a.id == id) {
                    active.total = total;
                    active.completed = 0;
                }
            }
            BatchEvent::ItemStarted { id, index } => {
                if self.active.as_ref().is_some_and(|a| a.id == id)
                    && let Some(item) = self.items.get_mut(index)
                {
                    item.state = ItemState::Processing;
                }
            }
            BatchEvent::ItemFinished { id, index, outcome } => {
                if let Some(active) = self.active.as_mut().filter(|a| a.id == id) {
                    active.completed += 1;
                    if let Some(item) = self.items.get_mut(index) {
                        item.state = match outcome {
                            ItemOutcome::Saved { output } => {
                                let _ = output;
                                ItemState::Saved
                            }
                            ItemOutcome::Skipped { output } => {
                                let _ = output;
                                ItemState::Skipped
                            }
                            ItemOutcome::Failed { error } => ItemState::Failed(error),
                        };
                    }
                }
            }
            BatchEvent::Finished { id, cancelled } => {
                if self.active.as_ref().is_some_and(|a| a.id == id) {
                    let active = self.active.take().expect("active batch checked");
                    let saved = self.items.iter().filter(|i| matches!(&i.state, ItemState::Saved)).count();
                    let skipped = self.items.iter().filter(|i| matches!(&i.state, ItemState::Skipped)).count();
                    let failed = self.items.iter().filter(|i| matches!(&i.state, ItemState::Failed(_))).count();
                    self.notice = Some(if cancelled {
                        format!("Cancelled · {}/{} completed · {saved} saved · {skipped} skipped · {failed} failed", active.completed, active.total)
                    } else {
                        format!("Finished · {saved} saved · {skipped} skipped · {failed} failed")
                    });
                }
            }
        }
    }

    pub fn model_downloaded(&mut self, model: &str, store: &ModelStore, worker: &Worker) {
        if self.waiting_for_model.is_some_and(|waiting| waiting == model) {
            self.waiting_for_model = None;
            self.launch(store, worker);
        }
    }

    pub fn model_download_failed(&mut self, model: &str) {
        if self.waiting_for_model.is_some_and(|waiting| waiting == model) {
            self.waiting_for_model = None;
            self.notice = Some(format!("Could not download model '{model}'."));
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, store: &ModelStore, worker: &mut Worker) {
        if ui.input(|i| i.viewport().close_requested()) {
            self.open = false;
            return;
        }

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.heading("Batch background removal");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(
                        self.active.is_none() && self.waiting_for_model.is_none(),
                        egui::Button::new("Clear queue"),
                    )
                    .clicked()
                {
                    self.items.clear();
                    self.notice = None;
                }
            });
        });
        ui.label("Process a list of images in the background. Results are saved as transparent PNGs.");
        ui.add_space(8.0);

        let busy = self.active.is_some() || self.waiting_for_model.is_some();
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!busy, |ui| {
                if ui.button("＋ Add files…").clicked() {
                    let files = rfd::FileDialog::new()
                        .add_filter("Images", IMAGE_EXTENSIONS)
                        .pick_files()
                        .unwrap_or_default();
                    self.add_files(files);
                }
                if ui.button("＋ Add folder…").clicked()
                    && let Some(folder) = rfd::FileDialog::new().pick_folder()
                {
                    self.add_folder(folder);
                }
                ui.checkbox(&mut self.recursive, "Include subfolders");
            });
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{} images queued", self.items.len())).strong());
            if let Some(active) = &self.active {
                ui.label(format!("· {} of {} completed", active.completed, active.total));
            }
        });
        egui::Frame::group(ui.style()).show(ui, |ui| {
            egui::ScrollArea::vertical()
                .max_height(230.0)
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    if self.items.is_empty() {
                        ui.add_space(28.0);
                        ui.vertical_centered(|ui| {
                            ui.label(RichText::new("Add images or choose a folder to begin").weak());
                        });
                        ui.add_space(28.0);
                    } else {
                        let mut remove = None;
                        for (index, item) in self.items.iter().enumerate() {
                            ui.horizontal(|ui| {
                                let (state, color, error) = item_state_label(&item.state);
                                ui.label(RichText::new(state).color(color).monospace().size(11.0));
                                let name = item.path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
                                let label = ui.label(name.to_string());
                                let tooltip = error.map_or_else(
                                    || item.path.display().to_string(),
                                    |message| format!("{}\n{message}", item.path.display()),
                                );
                                label.on_hover_text(tooltip);
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.add_enabled(!busy, egui::Button::new("×").frame(false)).clicked() {
                                        remove = Some(index);
                                    }
                                });
                            });
                        }
                        if let Some(index) = remove {
                            self.items.remove(index);
                        }
                    }
                });
        });

        ui.add_space(10.0);
        ui.heading("Output");
        ui.horizontal(|ui| {
            ui.label("Folder");
            ui.label(RichText::new(self.output_dir.display().to_string()).weak());
            if ui
                .add_enabled(!busy, egui::Button::new("Choose…"))
                .clicked()
                && let Some(folder) = rfd::FileDialog::new().set_directory(&self.output_dir).pick_folder()
            {
                self.output_dir = folder;
                self.output_dir_set = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Filename suffix");
            ui.add_enabled(!busy, egui::TextEdit::singleline(&mut self.suffix).desired_width(150.0));
            ui.add_enabled_ui(!busy, |ui| {
                ui.checkbox(&mut self.overwrite, "Overwrite existing outputs");
            });
        });
        ui.label(RichText::new("Example: photo.jpg → photo_nobg.png").small().weak());

        ui.add_space(8.0);
        ui.heading("Model");
        ui.add_enabled_ui(!busy, |ui| {
            egui::ComboBox::from_id_salt("batch-model")
                .selected_text(
                    registry::find(&self.selected_model)
                        .map(|m| {
                            format!(
                                "{}{}",
                                m.name,
                                if store.is_installed(m) { "" } else { " · download required" }
                            )
                        })
                        .unwrap_or_else(|| "Choose model".into()),
                )
                .show_ui(ui, |ui| {
                    for model in registry::MODELS {
                        let text = if store.is_installed(model) {
                            model.name.to_string()
                        } else {
                            format!("{} · download", model.name)
                        };
                        if ui.selectable_label(self.selected_model == model.id, text).clicked() {
                            self.selected_model = model.id.into();
                        }
                    }
                });
            if let Some(model) = registry::find(&self.selected_model)
                && !store.is_installed(model)
            {
                ui.label(RichText::new("This model will be downloaded before the batch starts.").small().weak());
            }
        });

        ui.add_space(10.0);
        ui.horizontal(|ui| {
            if self.active.is_none() && self.waiting_for_model.is_none() {
                let can_start = !self.items.is_empty() && !self.selected_model.is_empty();
                if ui
                    .add_enabled(can_start, egui::Button::new("▶ Start batch").fill(Color32::from_rgb(36, 125, 84)))
                    .clicked()
                {
                    self.request_start(store, worker);
                }
            } else if let Some(active) = &self.active {
                if ui.button("■ Cancel after current image").clicked() {
                    active.cancel.store(true, Ordering::Relaxed);
                    self.notice = Some("Cancellation requested. The current image will finish first.".into());
                }
                let fraction = if active.total == 0 {
                    0.0
                } else {
                    active.completed as f32 / active.total as f32
                };
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_width(190.0)
                        .text(format!("{}/{}", active.completed, active.total)),
                );
            } else {
                ui.spinner();
                ui.label("Downloading selected model…");
            }
        });

        if let Some(notice) = &self.notice {
            ui.add_space(5.0);
            ui.label(RichText::new(notice).color(ui.visuals().text_color()));
        }
        ui.add_space(6.0);
    }

    fn request_start(&mut self, store: &ModelStore, worker: &mut Worker) {
        if self.items.is_empty() {
            self.notice = Some("Add at least one image first.".into());
            return;
        }
        let Some(spec) = registry::find(&self.selected_model) else {
            self.notice = Some("Choose a valid model.".into());
            return;
        };

        if !store.is_installed(spec) {
            self.waiting_for_model = Some(spec.id);
            self.notice = Some(format!("Downloading {} before starting the batch…", spec.name));
            worker.download(spec);
            // Close a small race where the model finishes downloading between
            // the installed check above and Worker::download's deduplication.
            if store.is_installed(spec) {
                self.waiting_for_model = None;
                self.launch(store, worker);
            }
            return;
        }

        self.launch(store, worker);
    }

    fn launch(&mut self, store: &ModelStore, worker: &Worker) {
        let Some(spec) = registry::find(&self.selected_model) else { return };
        if !store.is_installed(spec) {
            self.notice = Some("The selected model is not installed yet.".into());
            return;
        }

        let files: Vec<PathBuf> = self.items.iter().map(|item| item.path.clone()).collect();
        if files.is_empty() {
            self.notice = Some("Add at least one image first.".into());
            return;
        }

        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.wrapping_add(1).max(1);
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        for item in &mut self.items {
            item.state = ItemState::Queued;
        }
        self.active = Some(ActiveBatch { id, cancel: cancel.clone(), total: files.len(), completed: 0 });
        self.notice = Some(format!("Starting {} image(s) with {}…", files.len(), spec.name));
        worker.run_batch(BatchJob {
            id,
            files,
            output_dir: self.output_dir.clone(),
            suffix: self.suffix.clone(),
            overwrite: self.overwrite,
            spec,
            cancel,
        });
    }

    fn add_files(&mut self, files: Vec<PathBuf>) {
        self.add_paths(files);
    }

    fn add_folder(&mut self, folder: PathBuf) {
        self.set_default_output_for(&folder);
        let mut files = Vec::new();
        let mut errors = 0usize;
        collect_images(&folder, self.recursive, &self.output_dir, &mut files, &mut errors);
        files.sort();
        self.add_paths(files);
        if errors > 0 {
            self.notice = Some(format!("Some folders could not be read ({errors} access error(s))."));
        }
    }

    fn add_paths(&mut self, paths: Vec<PathBuf>) {
        if let Some(first) = paths.first() {
            self.set_default_output_for(first.parent().unwrap_or_else(|| Path::new(".")));
        }
        let mut known: HashSet<PathBuf> = self.items.iter().map(|item| item.path.clone()).collect();
        let mut added = 0usize;
        for path in paths {
            if is_supported_image(&path) && path.is_file() && known.insert(path.clone()) {
                self.items.push(QueueItem { path, state: ItemState::Queued });
                added += 1;
            }
        }
        self.notice = Some(if added == 0 {
            "No new supported images were added.".into()
        } else {
            format!("Added {added} image(s).")
        });
    }

    fn set_default_output_for(&mut self, source: &Path) {
        if !self.output_dir_set {
            self.output_dir = source.join("Background Removed");
            self.output_dir_set = true;
        }
    }
}

fn item_state_label(state: &ItemState) -> (&'static str, Color32, Option<&str>) {
    match state {
        ItemState::Queued => ("Queued", Color32::GRAY, None),
        ItemState::Processing => ("Working", Color32::from_rgb(80, 150, 230), None),
        ItemState::Saved => ("Saved", Color32::from_rgb(70, 175, 115), None),
        ItemState::Skipped => ("Skipped", Color32::GRAY, None),
        ItemState::Failed(error) => ("Failed", Color32::from_rgb(220, 90, 80), Some(error)),
    }
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| IMAGE_EXTENSIONS.iter().any(|supported| ext.eq_ignore_ascii_case(supported)))
}

fn collect_images(
    folder: &Path,
    recursive: bool,
    excluded_output_dir: &Path,
    output: &mut Vec<PathBuf>,
    errors: &mut usize,
) {
    let entries = match fs::read_dir(folder) {
        Ok(entries) => entries,
        Err(_) => {
            *errors += 1;
            return;
        }
    };

    for entry in entries {
        let Ok(entry) = entry else {
            *errors += 1;
            continue;
        };
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            *errors += 1;
            continue;
        };
        if file_type.is_file() && is_supported_image(&path) {
            output.push(path);
        } else if recursive && file_type.is_dir() {
            // The default output directory lives inside the selected source
            // folder; do not recursively ingest results from an earlier run.
            if path.as_path() == excluded_output_dir || path.starts_with(excluded_output_dir) {
                continue;
            }
            collect_images(&path, recursive, excluded_output_dir, output, errors);
        }
    }
}
