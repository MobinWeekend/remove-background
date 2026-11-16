//! Native GUI (egui/eframe).

mod models_page;
mod view;
mod worker;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{
    Align, Color32, Key, KeyboardShortcut, Layout, Modifiers, RichText, TextureHandle, Ui, Vec2, vec2,
};
use image::{DynamicImage, RgbaImage};
use rembg_core::grid::{self, Tile, TileBg};
use rembg_core::{ActiveDevice, DevicePref, Engine, ModelSpec, ModelStore, imageops, registry};

use view::{Bg, Layer, Split, ViewState};
use worker::{DlState, Event, Output, Worker};

/// Initial tab when launching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StartTab {
    #[default]
    Remove,
    Compare,
    Models,
}

pub fn run(image: Option<PathBuf>, device: DevicePref, start: StartTab) -> anyhow::Result<()> {
    let store = ModelStore::open_default()?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("rembg")
            .with_app_id("rembg")
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([640.0, 420.0])
            .with_drag_and_drop(true),
        // Deliver Ctrl+C / Ctrl+V as plain key events: we handle image clipboard
        // ourselves (egui's own translation only covers text).
        clipboard_shortcuts: false,
        ..Default::default()
    };
    eframe::run_native(
        "rembg",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, store, device, image, start)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Remove,
    Compare,
    Models,
}

struct ModelResult {
    out: Arc<Output>,
    tex: TextureHandle,
    mask_tex: Option<TextureHandle>,
}

/// The currently loaded image and everything computed from it.
struct Doc {
    name: String,
    path: Option<PathBuf>,
    image: Arc<DynamicImage>,
    tex: TextureHandle,
    size: Vec2,
    job: u64,
    results: HashMap<&'static str, ModelResult>,
    pending: HashSet<&'static str>,
    running: Option<&'static str>,
    errors: HashMap<&'static str, String>,
}

struct Status {
    text: String,
    error: bool,
    at: Instant,
}

pub struct App {
    store: ModelStore,
    worker: Worker,
    tab: Tab,
    doc: Option<Doc>,
    model: &'static ModelSpec,
    default_model: &'static str,
    bg: Bg,
    custom_color: Color32,
    show_mask: bool,
    view: ViewState,
    checker: TextureHandle,
    clipboard: Option<arboard::Clipboard>,
    status: Option<Status>,
    device_pref: DevicePref,
    device: Option<ActiveDevice>,
    cuda_fallback: Option<String>,
    downloads: HashMap<&'static str, (u64, Option<u64>)>,
    /// Models to run once their download completes.
    run_after_download: HashSet<&'static str>,
    // Compare
    compare_models: HashSet<&'static str>,
    selected: Vec<&'static str>,
    split_mode: bool,
    split_pos: f32,
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        store: ModelStore,
        pref: DevicePref,
        image: Option<PathBuf>,
        start: StartTab,
    ) -> Self {
        let ctx = cc.egui_ctx.clone();
        let model = store.default_model();
        let worker = Worker::spawn(ctx.clone(), store.clone(), pref);
        let compare_models = store.installed().iter().map(|m| m.id).collect();
        let mut app = Self {
            checker: view::checker_texture(&ctx),
            default_model: model.id,
            store,
            worker,
            tab: match start {
                StartTab::Remove => Tab::Remove,
                StartTab::Compare => Tab::Compare,
                StartTab::Models => Tab::Models,
            },
            doc: None,
            model,
            bg: Bg::Checker,
            custom_color: Color32::from_rgb(0, 177, 64),
            show_mask: false,
            view: ViewState::default(),
            clipboard: None,
            status: None,
            device_pref: pref,
            device: None,
            cuda_fallback: None,
            downloads: HashMap::new(),
            run_after_download: HashSet::new(),
            compare_models,
            selected: Vec::new(),
            split_mode: false,
            split_pos: 0.5,
        };
        if let Some(p) = image {
            app.open_path(&ctx, p);
        }
        app
    }

    fn set_status(&mut self, text: impl Into<String>, error: bool) {
        let text = text.into();
        if error {
            tracing::error!("{text}");
        }
        self.status = Some(Status { text, error, at: Instant::now() });
    }

    // ---------- loading ----------

    fn open_path(&mut self, ctx: &egui::Context, path: PathBuf) {
        match imageops::load(&path) {
            Ok(img) => {
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.set_doc(ctx, img, name, Some(path));
            }
            Err(e) => self.set_status(format!("Could not open {}: {e}", path.display()), true),
        }
    }

    fn set_doc(&mut self, ctx: &egui::Context, img: DynamicImage, name: String, path: Option<PathBuf>) {
        let job = self.worker.new_job();
        let rgba = img.to_rgba8();
        let tex = view::load_texture(ctx, "source", &rgba);
        let size = vec2(img.width() as f32, img.height() as f32);
        self.set_status(format!("Loaded {name} ({}×{})", img.width(), img.height()), false);
        self.doc = Some(Doc {
            name,
            path,
            image: Arc::new(img),
            tex,
            size,
            job,
            results: HashMap::new(),
            pending: HashSet::new(),
            running: None,
            errors: HashMap::new(),
        });
        self.view = ViewState::default();
        self.selected.clear();
        self.split_mode = false;
        match self.tab {
            Tab::Compare => self.run_compare(),
            _ => self.run_model(self.model),
        }
    }

    fn paste(&mut self, ctx: &egui::Context) {
        let Some(cb) = self.clipboard() else { return };
        if let Ok(img) = cb.get_image() {
            let Some(rgba) = RgbaImage::from_raw(img.width as u32, img.height as u32, img.bytes.into_owned())
            else {
                return self.set_status("Clipboard image has an unexpected format", true);
            };
            return self.set_doc(ctx, DynamicImage::ImageRgba8(rgba), "pasted image".into(), None);
        }
        if let Ok(text) = cb.get_text() {
            let first = text.lines().next().unwrap_or("").trim();
            let p = PathBuf::from(first.strip_prefix("file://").unwrap_or(first));
            if p.is_file() {
                return self.open_path(ctx, p);
            }
        }
        self.set_status("Clipboard has no image", true);
    }

    fn open_dialog(&mut self, ctx: &egui::Context) {
        let picked = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"])
            .pick_file();
        if let Some(p) = picked {
            self.open_path(ctx, p);
        }
    }

    // ---------- running ----------

    fn run_model(&mut self, spec: &'static ModelSpec) {
        let Some(doc) = self.doc.as_mut() else { return };
        if doc.results.contains_key(spec.id) || doc.pending.contains(spec.id) {
            return;
        }
        doc.errors.remove(spec.id);
        if !self.store.is_installed(spec) {
            self.run_after_download.insert(spec.id);
            self.worker.download(spec);
            return;
        }
        doc.pending.insert(spec.id);
        self.worker.run(doc.job, spec, doc.image.clone());
    }

    fn run_compare(&mut self) {
        let mut specs: Vec<_> =
            registry::MODELS.iter().filter(|m| self.compare_models.contains(m.id)).collect();
        // Fast models first so the grid fills quickly.
        specs.sort_by_key(|m| (m.pre.size, m.size));
        for s in specs {
            self.run_model(s);
        }
    }

    fn select_model(&mut self, spec: &'static ModelSpec) {
        self.model = spec;
        self.run_model(spec);
    }

    // ---------- events ----------

    fn handle_events(&mut self, ctx: &egui::Context) {
        while let Ok(ev) = self.worker.rx.try_recv() {
            match ev {
                Event::Started { job, model } => {
                    if let Some(d) = self.doc.as_mut().filter(|d| d.job == job) {
                        d.running = Some(model);
                    }
                }
                Event::Done { job, model, out } => {
                    let Some(d) = self.doc.as_mut().filter(|d| d.job == job) else { continue };
                    d.pending.remove(model);
                    d.running = None;
                    let tex = view::load_texture(ctx, model, &out.cutout);
                    d.results.insert(model, ModelResult { out, tex, mask_tex: None });
                }
                Event::Failed { job, model, error } => {
                    if let Some(d) = self.doc.as_mut().filter(|d| d.job == job) {
                        d.pending.remove(model);
                        d.running = None;
                        d.errors.insert(model, error.clone());
                    }
                    self.set_status(format!("{model}: {error}"), true);
                }
                Event::Device { device, fallback } => {
                    self.device = Some(device);
                    self.cuda_fallback = fallback;
                }
                Event::Download { model, state } => match state {
                    DlState::Progress(done, total) => {
                        self.downloads.insert(model, (done, total));
                    }
                    DlState::Finished => {
                        self.downloads.remove(model);
                        self.worker.download_finished(model);
                        self.set_status(format!("Downloaded {model}"), false);
                        if self.run_after_download.remove(model)
                            && let Some(spec) = registry::find(model)
                        {
                            self.compare_models.insert(model);
                            self.run_model(spec);
                        }
                    }
                    DlState::Failed(e) => {
                        self.downloads.remove(model);
                        self.worker.download_finished(model);
                        self.run_after_download.remove(model);
                        self.set_status(format!("Download of {model} failed: {e}"), true);
                    }
                },
            }
        }
    }

    // ---------- output ----------

    fn clipboard(&mut self) -> Option<&mut arboard::Clipboard> {
        if self.clipboard.is_none() {
            match arboard::Clipboard::new() {
                Ok(c) => self.clipboard = Some(c),
                Err(e) => {
                    self.set_status(format!("Clipboard unavailable: {e}"), true);
                    return None;
                }
            }
        }
        self.clipboard.as_mut()
    }

    fn render_output(&self, model: &str, kind: OutKind) -> Option<RgbaImage> {
        let r = self.doc.as_ref()?.results.get(model)?;
        Some(match kind {
            OutKind::Transparent => r.out.cutout.clone(),
            OutKind::WithBg => match self.bg.solid() {
                Some(c) => DynamicImage::ImageRgb8(imageops::flatten(&r.out.cutout, c)).to_rgba8(),
                None => r.out.cutout.clone(),
            },
            OutKind::Mask => imageops::mask_rgba(&r.out.mask),
        })
    }

    fn copy(&mut self, model: &'static str, kind: OutKind) {
        let Some(img) = self.render_output(model, kind) else {
            return self.set_status("Nothing to copy yet", true);
        };
        let (w, h) = img.dimensions();
        let data = arboard::ImageData { width: w as usize, height: h as usize, bytes: img.into_raw().into() };
        let Some(cb) = self.clipboard() else { return };
        match cb.set_image(data) {
            Ok(()) => self.set_status(format!("Copied {w}×{h} {} to clipboard", kind.label()), false),
            Err(e) => self.set_status(format!("Copy failed: {e}"), true),
        }
    }

    fn save(&mut self, model: &'static str, kind: OutKind) {
        let Some(img) = self.render_output(model, kind) else {
            return self.set_status("Nothing to save yet", true);
        };
        let stem = self
            .doc
            .as_ref()
            .and_then(|d| d.path.as_ref())
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "image".into());
        let suffix = match kind {
            OutKind::Mask => "mask",
            _ => "nobg",
        };
        let mut dlg = rfd::FileDialog::new()
            .set_file_name(format!("{stem}_{suffix}.png"))
            .add_filter("PNG", &["png"])
            .add_filter("WebP", &["webp"])
            .add_filter("JPEG", &["jpg", "jpeg"]);
        if let Some(dir) = self.doc.as_ref().and_then(|d| d.path.as_ref()).and_then(|p| p.parent()) {
            dlg = dlg.set_directory(dir);
        }
        let Some(path) = dlg.save_file() else { return };
        let is_jpg = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg"));
        let res = if is_jpg {
            imageops::flatten(&img, self.bg.solid().unwrap_or([255; 3])).save(&path)
        } else {
            img.save(&path)
        };
        match res {
            Ok(()) => self.set_status(format!("Saved {}", path.display()), false),
            Err(e) => self.set_status(format!("Save failed: {e}"), true),
        }
    }

    fn save_grid(&mut self) {
        let Some(doc) = self.doc.as_ref() else { return };
        let mut entries: Vec<_> =
            registry::MODELS.iter().filter_map(|m| doc.results.get(m.id).map(|r| (m, r))).collect();
        entries.sort_by_key(|(m, _)| registry::MODELS.iter().position(|x| x.id == m.id));
        if entries.is_empty() {
            return self.set_status("Run the comparison first", true);
        }
        let masks: Vec<RgbaImage> = if self.show_mask {
            entries.iter().map(|(_, r)| imageops::mask_rgba(&r.out.mask)).collect()
        } else {
            Vec::new()
        };
        let tiles: Vec<Tile> = entries
            .iter()
            .enumerate()
            .map(|(i, (m, r))| Tile {
                label: format!("{} · {:.0} ms", m.name, r.out.infer.as_secs_f64() * 1e3),
                image: if self.show_mask { &masks[i] } else { &r.out.cutout },
            })
            .collect();
        let bg = match self.bg.solid() {
            Some(c) => TileBg::Solid(c),
            None => TileBg::Checker,
        };
        let stem = doc.path.as_ref().and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().into_owned());
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(format!("{}_compare.png", stem.unwrap_or_else(|| "image".into())))
            .add_filter("PNG", &["png"])
            .save_file()
        else {
            return;
        };
        match grid::compose(&tiles, 512, bg)
            .map_err(|e| e.to_string())
            .and_then(|g| g.save(&path).map_err(|e| e.to_string()))
        {
            Ok(()) => self.set_status(format!("Saved comparison to {}", path.display()), false),
            Err(e) => self.set_status(format!("Save failed: {e}"), true),
        }
    }

    fn set_default(&mut self, id: &'static str) {
        match self.store.set_default_model(id) {
            Ok(()) => {
                self.default_model = id;
                self.set_status(format!("Default model set to {id}"), false);
            }
            Err(e) => self.set_status(e.to_string(), true),
        }
    }

    // ---------- input ----------

    fn shortcuts(&mut self, ctx: &egui::Context) {
        let sc = |m, k| KeyboardShortcut::new(m, k);
        let (open, save, copy, mask, bg, reset) = ctx.input_mut(|i| {
            (
                i.consume_shortcut(&sc(Modifiers::COMMAND, Key::O)),
                i.consume_shortcut(&sc(Modifiers::COMMAND, Key::S)),
                i.consume_shortcut(&sc(Modifiers::COMMAND, Key::C)),
                i.consume_shortcut(&sc(Modifiers::NONE, Key::M)),
                i.consume_shortcut(&sc(Modifiers::NONE, Key::B)),
                i.consume_shortcut(&sc(Modifiers::NONE, Key::Num0)),
            )
        });
        // Ctrl+V arrives as a key event (clipboard_shortcuts is off); also accept
        // Paste / PasteImage in case a platform still emits them.
        let paste_key = ctx.input_mut(|i| i.consume_shortcut(&sc(Modifiers::COMMAND, Key::V)));
        let (paste_text, paste_image) = ctx.input(|i| {
            let text = i.events.iter().any(|e| matches!(e, egui::Event::Paste(_)));
            let img = i.events.iter().find_map(|e| match e {
                egui::Event::PasteImage(img) => Some(img.clone()),
                _ => None,
            });
            (text, img)
        });
        let paste = paste_key || paste_text;
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        if open {
            self.open_dialog(ctx);
        }
        if let Some(ci) = paste_image {
            let bytes: Vec<u8> = ci.pixels.iter().flat_map(|c| c.to_srgba_unmultiplied()).collect();
            if let Some(rgba) = RgbaImage::from_raw(ci.size[0] as u32, ci.size[1] as u32, bytes) {
                self.set_doc(ctx, DynamicImage::ImageRgba8(rgba), "pasted image".into(), None);
            }
        } else if paste {
            self.paste(ctx);
        }
        if copy {
            self.copy(self.model.id, OutKind::Transparent);
        }
        if save {
            self.save(self.model.id, OutKind::Transparent);
        }
        if mask {
            self.show_mask = !self.show_mask;
        }
        if bg {
            self.bg = match self.bg {
                Bg::Checker => Bg::Dark,
                Bg::Dark => Bg::Light,
                Bg::Light => Bg::Custom(self.custom_color),
                Bg::Custom(_) => Bg::Checker,
            };
        }
        if reset {
            self.view = ViewState::default();
        }
        let tab = ctx.input(|i| {
            if !i.modifiers.command {
                None
            } else if i.key_pressed(Key::Num1) {
                Some(Tab::Remove)
            } else if i.key_pressed(Key::Num2) {
                Some(Tab::Compare)
            } else if i.key_pressed(Key::Num3) {
                Some(Tab::Models)
            } else {
                None
            }
        });
        if let Some(t) = tab {
            self.switch_tab(t);
        }
    }

    fn handle_drop(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if let Some(f) = dropped.into_iter().next() {
            let path = f.path().to_path_buf();
            if path.is_file() {
                self.open_path(ctx, path);
            } else {
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                match f
                    .bytes()
                    .map_err(|e| e.to_string())
                    .and_then(|b| imageops::load_from_memory(&b).map_err(|e| e.to_string()))
                {
                    Ok(img) => self.set_doc(ctx, img, name, None),
                    Err(e) => self.set_status(format!("Could not open dropped file: {e}"), true),
                }
            }
        }
    }

    fn switch_tab(&mut self, t: Tab) {
        self.tab = t;
        if t == Tab::Compare {
            self.run_compare();
        }
    }

    // ---------- UI ----------

    fn top_bar(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            if ui.button("📂 Open").on_hover_text("Ctrl+O").clicked() {
                self.open_dialog(&ctx);
            }
            if ui.button("📋 Paste").on_hover_text("Ctrl+V").clicked() {
                self.paste(&ctx);
            }
            ui.separator();
            for (t, label) in
                [(Tab::Remove, "✂ Remove"), (Tab::Compare, "▦ Compare"), (Tab::Models, "⬇ Models")]
            {
                if ui.selectable_label(self.tab == t, label).clicked() {
                    self.switch_tab(t);
                }
            }
            if self.tab == Tab::Models {
                return;
            }
            ui.separator();
            if self.tab == Tab::Remove {
                self.model_combo(ui);
            }
            ui.label("Background");
            for (b, label, tip) in
                [(Bg::Checker, "▦", "Checkerboard"), (Bg::Dark, "⬛", "Dark"), (Bg::Light, "⬜", "Light")]
            {
                if ui.selectable_label(self.bg == b, label).on_hover_text(tip).clicked() {
                    self.bg = b;
                }
            }
            let custom = matches!(self.bg, Bg::Custom(_));
            if ui.selectable_label(custom, "🎨").on_hover_text("Custom colour").clicked() {
                self.bg = Bg::Custom(self.custom_color);
            }
            if custom && ui.color_edit_button_srgba(&mut self.custom_color).changed() {
                self.bg = Bg::Custom(self.custom_color);
            }
            ui.separator();
            ui.toggle_value(&mut self.show_mask, "◐ Mask").on_hover_text("Show the alpha mask (M)");

            if self.tab == Tab::Remove {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let id = self.model.id;
                    let ready = self.doc.as_ref().is_some_and(|d| d.results.contains_key(id));
                    ui.add_enabled_ui(ready, |ui| {
                        ui.menu_button("💾 Save ▾", |ui| {
                            if ui.button("Transparent PNG…").clicked() {
                                self.save(id, OutKind::Transparent);
                            }
                            if ui
                                .add_enabled(self.bg.solid().is_some(), egui::Button::new("With background…"))
                                .clicked()
                            {
                                self.save(id, OutKind::WithBg);
                            }
                            if ui.button("Mask…").clicked() {
                                self.save(id, OutKind::Mask);
                            }
                        });
                        ui.menu_button("▾", |ui| {
                            if ui
                                .add_enabled(
                                    self.bg.solid().is_some(),
                                    egui::Button::new("Copy with background"),
                                )
                                .clicked()
                            {
                                self.copy(id, OutKind::WithBg);
                            }
                            if ui.button("Copy mask").clicked() {
                                self.copy(id, OutKind::Mask);
                            }
                        });
                        if ui.button("📋 Copy").on_hover_text("Copy transparent result (Ctrl+C)").clicked()
                        {
                            self.copy(id, OutKind::Transparent);
                        }
                    });
                });
            }
        });
    }

    fn model_combo(&mut self, ui: &mut Ui) {
        let mut chosen = None;
        egui::ComboBox::from_id_salt("model").width(200.0).selected_text(self.model.name).show_ui(ui, |ui| {
            let mut family = None;
            for m in registry::MODELS {
                if family != Some(m.family) {
                    family = Some(m.family);
                    ui.label(RichText::new(m.family.label()).small().weak());
                }
                let installed = self.store.is_installed(m);
                let mut text = m.name.to_string();
                if !installed {
                    text.push_str("  ⬇");
                }
                if m.id == self.default_model {
                    text.push_str("  ★");
                }
                let tip = format!(
                    "{}\n{} · {:.0} MB · {}{}",
                    m.description,
                    m.id,
                    m.size as f64 / 1e6,
                    m.license,
                    if installed { "" } else { "\nNot installed — selecting downloads it" }
                );
                if ui.selectable_label(self.model.id == m.id, text).on_hover_text(tip).clicked() {
                    chosen = Some(m);
                }
            }
        });
        if let Some(m) = chosen {
            self.select_model(m);
        }
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if let Some(s) = &self.status {
                let color =
                    if s.error { Color32::from_rgb(240, 110, 100) } else { ui.visuals().text_color() };
                let fade = s.at.elapsed() < Duration::from_secs(8) || s.error;
                if fade {
                    ui.label(RichText::new(&s.text).color(color));
                }
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let dev = match self.device {
                    Some(d) => d.to_string(),
                    None => match self.device_pref {
                        DevicePref::Cpu => "CPU".into(),
                        _ if Engine::cuda_compiled() => "CUDA?".into(),
                        _ => "CPU".into(),
                    },
                };
                let r = ui.label(RichText::new(format!("⚙ {dev}")).monospace());
                if let Some(f) = &self.cuda_fallback {
                    r.on_hover_text(format!("CUDA fallback: {f}"));
                }
                if let Some(d) = &self.doc {
                    ui.separator();
                    ui.label(format!("{} · {}×{}", d.name, d.size.x, d.size.y));
                    if let Some(r) = d.results.get(self.model.id) {
                        ui.separator();
                        let load = r
                            .out
                            .load
                            .map(|l| format!(" (+{:.1}s load)", l.as_secs_f64()))
                            .unwrap_or_default();
                        ui.label(format!(
                            "{} {:.0} ms{load}",
                            self.model.id,
                            r.out.infer.as_secs_f64() * 1e3
                        ));
                    }
                }
                for (id, (done, total)) in &self.downloads {
                    ui.separator();
                    let frac = total.map(|t| *done as f32 / t as f32).unwrap_or(0.0);
                    ui.add(egui::ProgressBar::new(frac).desired_width(120.0).text(format!("⬇ {id}")));
                }
            });
        });
    }

    fn empty_state(&self, ui: &mut Ui) {
        ui.centered_and_justified(|ui| {
            ui.label(
                RichText::new("Drop an image here\n\nor paste (Ctrl+V) · open (Ctrl+O)")
                    .size(22.0)
                    .color(ui.visuals().weak_text_color()),
            );
        });
    }

    fn remove_tab(&mut self, ui: &mut Ui) {
        let Some(doc) = self.doc.as_mut() else { return self.empty_state(ui) };
        let rect = ui.available_rect_before_wrap();
        let id = self.model.id;
        let hold_original = ui.input(|i| i.key_down(Key::Space)) && !ui.ctx().egui_wants_keyboard_input();
        let ctx = ui.ctx().clone();
        let (tex, busy) = match doc.results.get_mut(id) {
            Some(r) if !hold_original => {
                if self.show_mask {
                    (mask_tex(&ctx, id, r).clone(), false)
                } else {
                    (r.tex.clone(), false)
                }
            }
            _ => (doc.tex.clone(), !doc.results.contains_key(id) && !hold_original),
        };
        view::canvas(
            ui,
            rect,
            egui::Id::unique("main-canvas"),
            Layer { tex: &tex },
            doc.size,
            &mut self.view,
            if busy { Bg::Dark } else { self.bg },
            &self.checker,
            None,
        );
        ui.allocate_rect(rect, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        if busy {
            let msg = if let Some(e) = doc.errors.get(id) {
                format!("⚠ {e}")
            } else if let Some((done, total)) = self.downloads.get(id) {
                format!(
                    "Downloading {id}… {:.0}%",
                    total.map(|t| *done as f64 / t as f64 * 100.0).unwrap_or(0.0)
                )
            } else if doc.running == Some(id) {
                format!("Removing background with {}…", self.model.name)
            } else {
                format!("Queued: {}", self.model.name)
            };
            overlay_label(&painter, rect, &msg);
            if !doc.errors.contains_key(id) {
                ui.ctx().request_repaint_after(Duration::from_millis(100));
                egui::Spinner::new().size(28.0).paint_at(
                    ui,
                    egui::Rect::from_center_size(rect.center() + vec2(0.0, -40.0), vec2(28.0, 28.0)),
                );
            }
        } else {
            let galley = painter.layout_no_wrap(
                "scroll: zoom · drag: pan · dbl-click: fit · hold Space: original · M: mask · B: background"
                    .into(),
                egui::FontId::proportional(11.0),
                Color32::from_white_alpha(200),
            );
            let pill = egui::Rect::from_min_size(
                rect.left_bottom() + vec2(8.0, -8.0 - galley.size().y - 6.0),
                galley.size() + vec2(12.0, 6.0),
            );
            painter.rect_filled(pill, 6.0, Color32::from_black_alpha(140));
            painter.galley(pill.min + vec2(6.0, 3.0), galley, Color32::WHITE);
        }
    }

    fn compare_tab(&mut self, ui: &mut Ui) {
        if self.doc.is_none() {
            return self.empty_state(ui);
        }
        // toolbar
        ui.horizontal(|ui| {
            ui.menu_button(format!("Models ({}) ▾", self.compare_models.len()), |ui| {
                ui.horizontal(|ui| {
                    if ui.small_button("all installed").clicked() {
                        self.compare_models = self.store.installed().iter().map(|m| m.id).collect();
                    }
                    if ui.small_button("none").clicked() {
                        self.compare_models.clear();
                    }
                });
                for m in registry::MODELS {
                    let installed = self.store.is_installed(m);
                    let mut on = self.compare_models.contains(m.id);
                    let label = if installed {
                        m.name.to_string()
                    } else {
                        format!("{}  ⬇ {:.0} MB", m.name, m.size as f64 / 1e6)
                    };
                    if ui.checkbox(&mut on, label).changed() {
                        if on {
                            self.compare_models.insert(m.id);
                        } else {
                            self.compare_models.remove(m.id);
                        }
                    }
                }
            });
            if ui.button("▶ Run").on_hover_text("Run all selected models (results are cached)").clicked() {
                self.run_compare();
            }
            ui.separator();
            let can_split = self.selected.len() == 2;
            ui.add_enabled_ui(can_split, |ui| {
                ui.toggle_value(&mut self.split_mode, "↔ Side by side")
                    .on_hover_text("Select two tiles (click) to compare with a slider");
            });
            if !can_split {
                self.split_mode = false;
                ui.label(RichText::new(format!("{}/2 selected", self.selected.len())).weak());
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button("💾 Save grid…").clicked() {
                    self.save_grid();
                }
            });
        });
        ui.separator();

        if self.split_mode {
            return self.split_view(ui);
        }

        let ctx = ui.ctx().clone();
        let doc = self.doc.as_mut().expect("checked");
        let models: Vec<&'static ModelSpec> = registry::MODELS
            .iter()
            .filter(|m| self.compare_models.contains(m.id) || doc.results.contains_key(m.id))
            .collect();
        if models.is_empty() {
            ui.label("Pick models to compare from the Models menu.");
            return;
        }
        let avail = ui.available_rect_before_wrap();
        let n = models.len();
        let aspect = doc.size.x / doc.size.y;
        let label_h = 26.0;
        // choose columns maximizing tile area
        let gap = 6.0;
        let (cols, tile) = (1..=n)
            .map(|c| {
                let rows = n.div_ceil(c);
                let w = (avail.width() - gap * (c as f32 - 1.0)) / c as f32;
                let h = (avail.height() - gap * (rows as f32 - 1.0)) / rows as f32 - label_h;
                let (tw, th) = if w / h > aspect { (h * aspect, h) } else { (w, w / aspect) };
                (c, vec2(tw.max(40.0), th.max(30.0)))
            })
            .max_by(|a, b| (a.1.x * a.1.y).total_cmp(&(b.1.x * b.1.y)))
            .unwrap();
        let cell = vec2((avail.width() - gap * (cols as f32 - 1.0)) / cols as f32, tile.y + label_h);

        let mut actions: Vec<(TileAction, &'static str)> = Vec::new();
        for (i, m) in models.iter().enumerate() {
            let (c, r) = (i % cols, i / cols);
            let min = avail.min + vec2(c as f32 * (cell.x + gap), r as f32 * (cell.y + gap));
            let header = egui::Rect::from_min_size(min, vec2(cell.x, label_h));
            let body = egui::Rect::from_min_size(min + vec2(0.0, label_h), vec2(cell.x, tile.y));
            let selected = self.selected.contains(&m.id);

            // header
            let mut hui = ui.new_child(
                egui::UiBuilder::new().max_rect(header).layout(Layout::left_to_right(Align::Center)),
            );
            let res = doc.results.get(m.id);
            let mut title = RichText::new(m.name).strong();
            if selected {
                title = title.color(Color32::from_rgb(80, 160, 255));
            }
            hui.label(title);
            if m.id == self.default_model {
                hui.label("★");
            }
            if let Some(r) = res {
                hui.label(
                    RichText::new(format!("{:.0} ms · {}", r.out.infer.as_secs_f64() * 1e3, r.out.device))
                        .monospace()
                        .weak(),
                );
                hui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let resp = ui.small_button("Use").on_hover_text("Open in Remove tab with this model");
                    if resp.clicked() {
                        actions.push((TileAction::Use, m.id));
                    }
                    if ui.small_button("📋").on_hover_text("Copy result").clicked() {
                        actions.push((TileAction::Copy, m.id));
                    }
                    if m.id != self.default_model
                        && ui.small_button("★").on_hover_text("Set as default model").clicked()
                    {
                        actions.push((TileAction::Default, m.id));
                    }
                });
            }

            // body
            match doc.results.get_mut(m.id) {
                Some(r) => {
                    let tex = if self.show_mask { mask_tex(&ctx, m.id, r).clone() } else { r.tex.clone() };
                    let resp = view::canvas(
                        ui,
                        body,
                        egui::Id::unique(("tile", m.id)),
                        Layer { tex: &tex },
                        doc.size,
                        &mut self.view,
                        self.bg,
                        &self.checker,
                        None,
                    );
                    if resp.clicked() {
                        actions.push((TileAction::ToggleSelect, m.id));
                    }
                    if selected {
                        ui.painter().rect_stroke(
                            body,
                            2.0,
                            egui::Stroke::new(2.5, Color32::from_rgb(80, 160, 255)),
                            egui::StrokeKind::Inside,
                        );
                    }
                }
                None => {
                    ui.painter().rect_filled(body, 4.0, ui.visuals().extreme_bg_color);
                    let msg = if let Some(e) = doc.errors.get(m.id) {
                        format!("⚠ {e}")
                    } else if let Some((d, t)) = self.downloads.get(m.id) {
                        format!("Downloading… {:.0}%", t.map(|t| *d as f64 / t as f64 * 100.0).unwrap_or(0.0))
                    } else if doc.running == Some(m.id) {
                        "Running…".into()
                    } else if doc.pending.contains(m.id) {
                        "Queued".into()
                    } else if !self.store.is_installed(m) {
                        "Not installed — press Run to download".into()
                    } else {
                        "Press Run".into()
                    };
                    overlay_label(&ui.painter_at(body), body, &msg);
                    if doc.running == Some(m.id) || self.downloads.contains_key(m.id) {
                        ui.ctx().request_repaint_after(Duration::from_millis(100));
                        egui::Spinner::new().size(20.0).paint_at(
                            ui,
                            egui::Rect::from_center_size(body.center() + vec2(0.0, -28.0), vec2(20.0, 20.0)),
                        );
                    }
                }
            }
        }
        ui.allocate_rect(avail, egui::Sense::hover());

        for (a, id) in actions {
            match a {
                TileAction::ToggleSelect => {
                    if let Some(p) = self.selected.iter().position(|s| *s == id) {
                        self.selected.remove(p);
                    } else {
                        self.selected.push(id);
                        if self.selected.len() > 2 {
                            self.selected.remove(0);
                        }
                    }
                }
                TileAction::Use => {
                    if let Some(m) = registry::find(id) {
                        self.model = m;
                        self.tab = Tab::Remove;
                    }
                }
                TileAction::Copy => self.copy(id, OutKind::Transparent),
                TileAction::Default => self.set_default(id),
            }
        }
    }

    fn split_view(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        let [a, b] = [self.selected[0], self.selected[1]];
        let doc = self.doc.as_mut().expect("checked");
        let tex_of = |doc: &mut Doc, id: &'static str, mask: bool| -> Option<TextureHandle> {
            let r = doc.results.get_mut(id)?;
            Some(if mask { mask_tex(&ctx, id, r).clone() } else { r.tex.clone() })
        };
        let (Some(ta), Some(tb)) = (tex_of(doc, a, self.show_mask), tex_of(doc, b, self.show_mask)) else {
            self.split_mode = false;
            return;
        };
        let name = |id| registry::find(id).map(|m| m.name).unwrap_or(id);
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("◀ {}", name(a))).strong());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(RichText::new(format!("{} ▶", name(b))).strong());
                ui.label(RichText::new("drag the divider · scroll to zoom").weak());
            });
        });
        let rect = ui.available_rect_before_wrap();
        view::canvas(
            ui,
            rect,
            egui::Id::unique("split-canvas"),
            Layer { tex: &ta },
            doc.size,
            &mut self.view,
            self.bg,
            &self.checker,
            Some(Split { right: &tb, pos: &mut self.split_pos }),
        );
        ui.allocate_rect(rect, egui::Sense::hover());
    }
}

#[derive(Clone, Copy)]
enum TileAction {
    ToggleSelect,
    Use,
    Copy,
    Default,
}

#[derive(Clone, Copy)]
enum OutKind {
    Transparent,
    WithBg,
    Mask,
}

impl OutKind {
    fn label(self) -> &'static str {
        match self {
            OutKind::Transparent => "cut-out",
            OutKind::WithBg => "image",
            OutKind::Mask => "mask",
        }
    }
}

fn mask_tex<'a>(ctx: &egui::Context, id: &str, r: &'a mut ModelResult) -> &'a TextureHandle {
    r.mask_tex.get_or_insert_with(|| {
        view::load_texture(ctx, &format!("{id}-mask"), &imageops::mask_rgba(&r.out.mask))
    })
}

fn overlay_label(painter: &egui::Painter, rect: egui::Rect, text: &str) {
    let galley = painter.layout(
        text.to_owned(),
        egui::FontId::proportional(15.0),
        Color32::WHITE,
        (rect.width() - 40.0).max(80.0),
    );
    let r = egui::Rect::from_center_size(rect.center(), galley.size() + vec2(24.0, 14.0));
    painter.rect_filled(r, 8.0, Color32::from_black_alpha(170));
    painter.galley(r.min + vec2(12.0, 7.0), galley, Color32::WHITE);
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_events(ctx);
        self.handle_drop(ctx);
        self.shortcuts(ctx);
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(2.0);
        });
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        egui::CentralPanel::default_margins().show(ui, |ui| match self.tab {
            Tab::Remove => self.remove_tab(ui),
            Tab::Compare => self.compare_tab(ui),
            Tab::Models => models_page::show(self, ui),
        });

        // drop overlay
        let hovering = ui.input(|i| !i.raw.hovered_files.is_empty());
        if hovering {
            let rect = ui.ctx().content_rect();
            let painter =
                ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("drop")));
            painter.rect_filled(rect, 0.0, Color32::from_black_alpha(160));
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drop to remove background",
                egui::FontId::proportional(28.0),
                Color32::WHITE,
            );
        }
    }
}
