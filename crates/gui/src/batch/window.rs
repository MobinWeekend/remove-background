//! Secondary viewport and controls for batch background removal.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use egui::{Color32, RichText, TextureHandle, vec2};
use image::RgbaImage;
use rembg_core::{ModelStore, imageops, registry};

use crate::{view, worker::Worker};

use super::{BatchEvent, BatchJob, ItemOutcome, processor};

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"];
const PREVIEW_MAX_SIDE: u32 = 768;

struct QueueItem {
    path: PathBuf,
    output_path: PathBuf,
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

#[derive(Clone, PartialEq, Eq)]
struct PreviewKey {
    input: PathBuf,
    output: PathBuf,
}

struct PreviewImage {
    texture: TextureHandle,
    dimensions: [u32; 2],
}

struct PreviewLoadResult {
    before: Result<(RgbaImage, [u32; 2]), String>,
    after: Result<Option<(RgbaImage, [u32; 2])>, String>,
}

struct PreviewCache {
    key: PreviewKey,
    receiver: Option<mpsc::Receiver<PreviewLoadResult>>,
    before: Option<PreviewImage>,
    after: Option<PreviewImage>,
    before_error: Option<String>,
    after_error: Option<String>,
    loading: bool,
}

pub struct BatchWindow {
    pub open: bool,
    items: Vec<QueueItem>,
    selected_item: Option<PathBuf>,
    preview: Option<PreviewCache>,
    output_dir: PathBuf,
    output_dir_set: bool,
    recursive: bool,
    suffix: String,
    skip_existing: bool,
    selected_model: String,
    next_job_id: u64,
    active: Option<ActiveBatch>,
    waiting_for_model: Option<&'static str>,
    notice: Option<String>,
    /// Queue and before-preview widths; after-preview receives the remaining space.
    column_widths: [f32; 2],
    column_drag_start: Option<(usize, f32)>,
}

impl BatchWindow {
    pub fn new(default_model: &str) -> Self {
        Self {
            open: false,
            items: Vec::new(),
            selected_item: None,
            preview: None,
            output_dir: std::env::current_dir()
                .unwrap_or_default()
                .join("Background Removed"),
            output_dir_set: false,
            recursive: false,
            suffix: "_nobg".into(),
            skip_existing: true,
            selected_model: default_model.into(),
            next_job_id: 1,
            active: None,
            waiting_for_model: None,
            notice: None,
            column_widths: [270.0, 300.0],
            column_drag_start: None,
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
                let matching_batch = self.active.as_ref().is_some_and(|a| a.id == id);
                if matching_batch {
                    if let Some(active) = self.active.as_mut() {
                        active.completed += 1;
                    }
                    if let Some(item) = self.items.get_mut(index) {
                        item.state = match outcome {
                            ItemOutcome::Saved { output } => {
                                item.output_path = output;
                                ItemState::Saved
                            }
                            ItemOutcome::Skipped { output } => {
                                item.output_path = output;
                                ItemState::Skipped
                            }
                            ItemOutcome::Failed { error } => ItemState::Failed(error),
                        };
                        if self.selected_item.as_ref() == Some(&item.path) {
                            self.preview = None;
                        }
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
                        format!(
                            "Cancelled · {}/{} completed · {saved} saved · {skipped} skipped · {failed} failed",
                            active.completed, active.total
                        )
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

        let dropped_files = ui.input(|i| i.raw.dropped_files.clone());
        let busy = self.active.is_some() || self.waiting_for_model.is_some();
        if !dropped_files.is_empty() {
            if busy {
                self.notice = Some("Wait for the current batch to finish before adding more files.".into());
            } else {
                let paths = dropped_files.into_iter().map(|file| file.path().to_path_buf()).collect();
                self.add_sources(paths);
            }
        }

        ui.add_space(8.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading(RichText::new("Batch background removal").strong());
                    ui.label(RichText::new("Process a folder or selection into transparent PNGs").weak().small());
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(
                            !busy && !self.items.is_empty(),
                            egui::Button::new(RichText::new("Clear queue").size(13.0))
                                .fill(if ui.visuals().dark_mode {
                                    Color32::from_rgb(67, 55, 58)
                                } else {
                                    Color32::from_rgb(244, 226, 228)
                                })
                                .min_size(vec2(104.0, 34.0)),
                        )
                        .clicked()
                    {
                        self.items.clear();
                        self.selected_item = None;
                        self.preview = None;
                        self.notice = None;
                    }
                });
            });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!busy, |ui| {
                    let dark_mode = ui.visuals().dark_mode;
                    let add_button = |text| {
                        egui::Button::new(RichText::new(text).strong().size(14.0))
                            .fill(if dark_mode {
                                Color32::from_rgb(48, 88, 108)
                            } else {
                                Color32::from_rgb(218, 237, 244)
                            })
                            .min_size(vec2(128.0, 36.0))
                    };
                    if ui.add(add_button("+ Add files…")).clicked() {
                        let files = rfd::FileDialog::new()
                            .add_filter("Images", IMAGE_EXTENSIONS)
                            .pick_files()
                            .unwrap_or_default();
                        self.add_sources(files);
                    }
                    if ui.add(add_button("+ Add folder…")).clicked()
                        && let Some(folder) = rfd::FileDialog::new().pick_folder()
                    {
                        self.add_sources(vec![folder]);
                    }
                    ui.add_space(4.0);
                    ui.checkbox(&mut self.recursive, "Include subfolders");
                });
            });
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Queue · {} image(s)", self.items.len())).strong());
            if let Some(active) = &self.active {
                ui.label(RichText::new(format!("{} of {} completed", active.completed, active.total)).weak());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new("Drag the dividers to resize List / Before / After").small().weak());
            });
        });

        let ctx = ui.ctx().clone();
        let total_width = ui.available_width();
        let splitter_width = 10.0;
        let content_width = (total_width - splitter_width * 2.0).max(560.0);
        let queue_width = self.column_widths[0].clamp(210.0, (content_width - 360.0).max(210.0));
        let before_width = self.column_widths[1]
            .clamp(180.0, (content_width - queue_width - 180.0).max(180.0));
        self.column_widths = [queue_width, before_width];
        let after_width = (content_width - queue_width - before_width).max(180.0);
        let content_height = 326.0;

        ui.horizontal(|row| {
            row.spacing_mut().item_spacing.x = 0.0;
            row.allocate_ui_with_layout(
                vec2(queue_width, content_height),
                egui::Layout::top_down(egui::Align::Min),
                |column| self.show_queue(column, busy),
            );
            self.show_column_splitter(row, 0, content_width, content_height);
            row.allocate_ui_with_layout(
                vec2(before_width, content_height),
                egui::Layout::top_down(egui::Align::Min),
                |column| self.show_preview_side(column, &ctx, false),
            );
            self.show_column_splitter(row, 1, content_width, content_height);
            row.allocate_ui_with_layout(
                vec2(after_width, content_height),
                egui::Layout::top_down(egui::Align::Min),
                |column| self.show_preview_side(column, &ctx, true),
            );
        });

        ui.add_space(8.0);
        ui.separator();
        ui.heading("Output");
        let mut output_settings_changed = false;
        ui.horizontal(|ui| {
            ui.label("Folder");
            let folder_text = self.output_dir.display().to_string();
            ui.label(RichText::new(folder_text.clone()).weak()).on_hover_text(folder_text);
            if ui
                .add_enabled(!busy, egui::Button::new("Choose…"))
                .clicked()
                && let Some(folder) = rfd::FileDialog::new().set_directory(&self.output_dir).pick_folder()
            {
                self.output_dir = folder;
                self.output_dir_set = true;
                output_settings_changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Filename suffix");
            output_settings_changed |= ui
                .add_enabled(!busy, egui::TextEdit::singleline(&mut self.suffix).desired_width(130.0))
                .changed();
            ui.add_enabled_ui(!busy, |ui| {
                ui.checkbox(&mut self.skip_existing, "Skip files that already exist");
            });
        });
        ui.label(RichText::new("Example: photo.jpg → photo_nobg.png").small().weak());
        if output_settings_changed {
            self.update_output_paths();
            self.preview = None;
        }

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Model");
            ui.add_enabled_ui(!busy, |ui| {
                egui::ComboBox::from_id_salt("batch-model")
                    .selected_text(
                        registry::find(&self.selected_model)
                            .map(|model| {
                                format!(
                                    "{}{}",
                                    model.name,
                                    if store.is_installed(model) { "" } else { " · download required" }
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
            });
            if let Some(model) = registry::find(&self.selected_model)
                && !store.is_installed(model)
            {
                ui.label(RichText::new("Download required before starting").small().weak());
            }
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if self.active.is_none() && self.waiting_for_model.is_none() {
                let can_start = !self.items.is_empty() && !self.selected_model.is_empty();
                if ui
                    .add_enabled(
                        can_start,
                        egui::Button::new(RichText::new("▶  Start batch").strong().size(15.0))
                            .fill(Color32::from_rgb(32, 132, 91))
                            .min_size(vec2(164.0, 40.0)),
                    )
                    .clicked()
                {
                    self.request_start(store, worker);
                }
            } else if let Some(active) = &self.active {
                if ui
                    .add(
                        egui::Button::new(RichText::new("■  Cancel after current image").strong())
                            .fill(Color32::from_rgb(145, 65, 65))
                            .min_size(vec2(210.0, 38.0)),
                    )
                    .clicked()
                {
                    active.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    self.notice = Some("Cancellation requested. The current image will finish first.".into());
                }
                let fraction = if active.total == 0 {
                    0.0
                } else {
                    active.completed as f32 / active.total as f32
                };
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_width(220.0)
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

        if ui.input(|i| !i.raw.hovered_files.is_empty()) {
            let rect = ui.max_rect();
            let painter = ui.ctx().layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                ui.id().with("batch-drop-overlay"),
            ));
            painter.rect_filled(rect, 0.0, Color32::from_black_alpha(150));
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drop images or folders to add to the batch",
                egui::FontId::proportional(24.0),
                Color32::WHITE,
            );
        }
    }

    fn show_queue(&mut self, ui: &mut egui::Ui, busy: bool) {
        ui.label(RichText::new("File list").strong().size(14.0));
        egui::Frame::group(ui.style()).show(ui, |ui| {
            egui::ScrollArea::vertical()
                .max_height(286.0)
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    if self.items.is_empty() {
                        ui.add_space(62.0);
                        ui.vertical_centered(|ui| {
                            ui.label(RichText::new("No images added yet").weak());
                            ui.label(RichText::new("Drop files or folders here").small().weak());
                        });
                        ui.add_space(62.0);
                        return;
                    }

                    let mut remove_index = None;
                    let mut selected_path = None;
                    for (index, item) in self.items.iter().enumerate() {
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(!busy, egui::Button::new("×").frame(false).small())
                                .on_hover_text("Remove from queue")
                                .clicked()
                            {
                                remove_index = Some(index);
                            }

                            let name = item
                                .path
                                .file_name()
                                .map(|name| name.to_string_lossy().into_owned())
                                .unwrap_or_else(|| item.path.display().to_string());
                            let selected = self.selected_item.as_ref() == Some(&item.path);
                            let (state, color, error) = item_state_label(&item.state);
                            let response = ui.selectable_label(selected, name).on_hover_text(error.map_or_else(
                                || format!("{}\n{state}", item.path.display()),
                                |message| format!("{}\n{state}: {message}", item.path.display()),
                            ));
                            if response.clicked() {
                                selected_path = Some(item.path.clone());
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(state).color(color).monospace().size(10.0));
                            });
                        });
                    }

                    if let Some(path) = selected_path {
                        if self.selected_item.as_ref() != Some(&path) {
                            self.selected_item = Some(path);
                            self.preview = None;
                        }
                    }
                    if let Some(index) = remove_index {
                        let removed = self.items.remove(index).path;
                        if self.selected_item.as_ref() == Some(&removed) {
                            self.selected_item = self
                                .items
                                .get(index.min(self.items.len().saturating_sub(1)))
                                .map(|item| item.path.clone());
                        }
                        self.preview = None;
                        self.update_output_paths();
                    }
                });
        });
    }

    fn show_column_splitter(
        &mut self,
        ui: &mut egui::Ui,
        column: usize,
        content_width: f32,
        height: f32,
    ) {
        let (rect, response) = ui.allocate_exact_size(vec2(10.0, height), egui::Sense::click_and_drag());
        if response.drag_started() {
            self.column_drag_start = Some((column, self.column_widths[column]));
        }
        if response.dragged()
            && let Some((active_column, start_width)) = self.column_drag_start
            && active_column == column
        {
            let min_width = if column == 0 { 210.0 } else { 180.0 };
            let max_width = if column == 0 {
                content_width - 360.0
            } else {
                content_width - self.column_widths[0] - 180.0
            };
            if let Some(delta) = response.total_drag_delta() {
                self.column_widths[column] = (start_width + delta.x)
                    .clamp(min_width, max_width.max(min_width));
            }
        }
        if response.drag_stopped() {
            self.column_drag_start = None;
        }
        if response.hovered() || response.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }

        let color = if response.dragged() {
            Color32::from_rgb(55, 158, 190)
        } else if response.hovered() {
            Color32::from_rgb(90, 135, 153)
        } else if ui.visuals().dark_mode {
            Color32::from_gray(70)
        } else {
            Color32::from_gray(190)
        };
        let handle = rect.shrink2(vec2(3.0, 8.0));
        ui.painter().rect_filled(handle, 4.0, color);
        for offset in [-7.0, 0.0, 7.0] {
            ui.painter().circle_filled(rect.center() + vec2(0.0, offset), 1.25, Color32::from_white_alpha(180));
        }
        response.on_hover_text("Drag to resize this column");
    }

    fn show_preview_side(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, after: bool) {
        let title = if after { "After" } else { "Before" };
        let selected = self.selected_item.as_ref().and_then(|path| {
            self.items
                .iter()
                .find(|item| &item.path == path)
                .map(|item| {
                    (
                        PreviewKey { input: item.path.clone(), output: item.output_path.clone() },
                        matches!(&item.state, ItemState::Queued | ItemState::Processing),
                    )
                })
        });
        let Some((key, waiting_for_result)) = selected else {
            show_preview_tile(
                ui,
                title,
                None,
                None,
                "Select an image from the list",
            );
            return;
        };

        self.ensure_preview(ctx, key);
        let (image, error, loading) = self.preview.as_ref().map_or((None, None, true), |preview| {
            if after {
                (preview.after.as_ref(), preview.after_error.as_deref(), preview.loading)
            } else {
                (preview.before.as_ref(), preview.before_error.as_deref(), preview.loading)
            }
        });
        let placeholder = if loading {
            "Loading preview…"
        } else if after && waiting_for_result {
            "Result appears after processing"
        } else if after {
            "No output available"
        } else {
            "Could not load source image"
        };
        show_preview_tile(ui, title, image, error, placeholder);
    }

    fn ensure_preview(&mut self, ctx: &egui::Context, key: PreviewKey) {
        let needs_load = self.preview.as_ref().is_none_or(|preview| preview.key != key);
        if needs_load {
            let (sender, receiver) = mpsc::channel();
            let input = key.input.clone();
            let output = key.output.clone();
            let spawn = std::thread::Builder::new()
                .name("batch-preview".into())
                .spawn(move || {
                    let before = load_preview_pixels(&input);
                    let after = if output.is_file() {
                        load_preview_pixels(&output).map(Some)
                    } else {
                        Ok(None)
                    };
                    let _ = sender.send(PreviewLoadResult { before, after });
                });
            self.preview = Some(PreviewCache {
                key,
                receiver: Some(receiver),
                before: None,
                after: None,
                before_error: None,
                after_error: None,
                loading: spawn.is_ok(),
            });
            if let Err(error) = spawn
                && let Some(preview) = self.preview.as_mut()
            {
                preview.receiver = None;
                preview.before_error = Some(format!("Could not start preview loader: {error}"));
            }
        }

        let loaded = self
            .preview
            .as_ref()
            .and_then(|preview| preview.receiver.as_ref())
            .and_then(|receiver| receiver.try_recv().ok());
        if let Some(loaded) = loaded
            && let Some(preview) = self.preview.as_mut()
        {
            preview.receiver = None;
            preview.loading = false;
            match loaded.before {
                Ok((pixels, dimensions)) => {
                    preview.before = Some(PreviewImage {
                        texture: view::load_texture(ctx, "batch-preview-before", &pixels),
                        dimensions,
                    });
                }
                Err(error) => preview.before_error = Some(error),
            }
            match loaded.after {
                Ok(Some((pixels, dimensions))) => {
                    preview.after = Some(PreviewImage {
                        texture: view::load_texture(ctx, "batch-preview-after", &pixels),
                        dimensions,
                    });
                }
                Ok(None) => {}
                Err(error) => preview.after_error = Some(error),
            }
        }
        if self.preview.as_ref().is_some_and(|preview| preview.loading) {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
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
            self.notice = Some(format!("Downloading {} before the batch starts…", spec.name));
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

        self.update_output_paths();
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
            skip_existing: self.skip_existing,
            spec,
            cancel,
        });
    }

    /// Accept both files and folders so file-dialog and OS drag/drop share one path.
    fn add_sources(&mut self, paths: Vec<PathBuf>) {
        let mut files = Vec::new();
        let mut errors = 0usize;
        for path in paths {
            if path.is_dir() {
                self.set_default_output_for(&path);
                collect_images(&path, self.recursive, &self.output_dir, &mut files, &mut errors);
            } else {
                files.push(path);
            }
        }
        files.sort();
        let added = self.insert_paths(files);
        self.notice = Some(if errors > 0 {
            format!("Added {added} image(s); {errors} folder access error(s).")
        } else if added == 0 {
            "No new supported images were added.".into()
        } else {
            format!("Added {added} image(s).")
        });
    }

    fn insert_paths(&mut self, paths: Vec<PathBuf>) -> usize {
        if let Some(first) = paths.first() {
            self.set_default_output_for(first.parent().unwrap_or_else(|| Path::new(".")));
        }
        let mut known: HashSet<PathBuf> = self.items.iter().map(|item| item.path.clone()).collect();
        let mut added_paths = Vec::new();
        for path in paths {
            if is_supported_image(&path) && path.is_file() && known.insert(path.clone()) {
                added_paths.push(path);
            }
        }
        let added = added_paths.len();
        self.items.extend(added_paths.into_iter().map(|path| QueueItem {
            path,
            output_path: PathBuf::new(),
            state: ItemState::Queued,
        }));
        if self.selected_item.is_none() {
            self.selected_item = self.items.first().map(|item| item.path.clone());
        }
        self.update_output_paths();
        added
    }

    fn update_output_paths(&mut self) {
        let files: Vec<PathBuf> = self.items.iter().map(|item| item.path.clone()).collect();
        let outputs = processor::output_paths(&files, &self.output_dir, &self.suffix);
        for (item, output) in self.items.iter_mut().zip(outputs) {
            item.output_path = output;
        }
    }

    fn set_default_output_for(&mut self, source: &Path) {
        if !self.output_dir_set {
            self.output_dir = source.join("Background Removed");
            self.output_dir_set = true;
        }
    }
}

fn show_preview_tile(
    ui: &mut egui::Ui,
    title: &str,
    image: Option<&PreviewImage>,
    error: Option<&str>,
    placeholder: &str,
) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_min_height(300.0);
        ui.vertical_centered(|ui| {
            let title_color = match title {
                "Before" => Color32::from_rgb(93, 165, 194),
                "After" => Color32::from_rgb(92, 180, 132),
                _ => ui.visuals().text_color(),
            };
            ui.label(RichText::new(title).strong().size(15.0).color(title_color));
            ui.add_space(5.0);
            if let Some(image) = image {
                let available = ui.available_size();
                let source = vec2(image.dimensions[0] as f32, image.dimensions[1] as f32);
                let scale = (available.x / source.x.max(1.0))
                    .min((available.y - 30.0).max(1.0) / source.y.max(1.0))
                    .min(1.0);
                let size = source * scale;
                ui.image((image.texture.id(), size));
            } else {
                if let Some(error) = error {
                    ui.label(RichText::new(error).color(Color32::from_rgb(220, 90, 80)).small());
                } else {
                    ui.add_space(78.0);
                    ui.label(RichText::new(placeholder).weak().small());
                }
            }
        });
    });
}

fn load_preview_pixels(path: &Path) -> Result<(RgbaImage, [u32; 2]), String> {
    let image = imageops::load(path).map_err(|error| format!("Could not read {}: {error}", path.display()))?;
    let dimensions = [image.width(), image.height()];
    let fitted = imageops::fit_rgba(&image.to_rgba8(), PREVIEW_MAX_SIDE)
        .map_err(|error| format!("Could not prepare preview: {error}"))?;
    Ok((imageops::over_checker(&fitted, 12), dimensions))
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
