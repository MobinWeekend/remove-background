//! Model manager: install state, sizes, licences, download/delete/default, device.

use egui::{Color32, RichText, Ui};
use rembg_core::{DevicePref, Engine, registry};

use crate::App;

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 { format!("{:.2} GB", b / 1e9) } else { format!("{:.0} MB", b / 1e6) }
}

/// Display a path with the home directory abbreviated to `~`.
fn tilde(p: &std::path::Path) -> String {
    match std::env::var_os("HOME").map(std::path::PathBuf::from) {
        Some(home) if p.starts_with(&home) => {
            format!("~/{}", p.strip_prefix(&home).unwrap_or(p).display())
        }
        _ => p.display().to_string(),
    }
}

pub fn show(app: &mut App, ui: &mut Ui) {
    let installed = app.store.installed();
    let used: u64 = installed.iter().filter_map(|m| app.store.disk_size(m)).sum();
    let total: u64 = registry::MODELS.iter().map(|m| m.size).sum();

    ui.horizontal(|ui| {
        ui.heading("Models");
        ui.label(
            RichText::new(format!(
                "{} of {} installed · {} on disk · {} for all",
                installed.len(),
                registry::MODELS.len(),
                human(used),
                human(total)
            ))
            .weak(),
        );
    });
    ui.horizontal(|ui| {
        ui.label(RichText::new(tilde(&app.store.models_dir())).monospace().small());
        if ui.small_button("Open folder").clicked()
            && let Err(e) = std::process::Command::new("xdg-open").arg(app.store.models_dir()).spawn()
        {
            app.set_status(format!("xdg-open failed: {e}"), true);
        }
    });
    ui.add_space(6.0);

    // Device
    ui.horizontal(|ui| {
        ui.label(RichText::new("Device").strong());
        let before = app.device_pref;
        ui.selectable_value(&mut app.device_pref, DevicePref::Auto, "Auto");
        ui.selectable_value(&mut app.device_pref, DevicePref::Cpu, "CPU");
        ui.add_enabled_ui(Engine::cuda_compiled(), |ui| {
            ui.selectable_value(&mut app.device_pref, DevicePref::Cuda, "CUDA")
                .on_disabled_hover_text("This build has no CUDA support (make cuda)");
        });
        if app.device_pref != before {
            app.worker.set_device(app.device_pref);
            app.device = None;
            app.cuda_fallback = None;
        }
        let active = app.device.map(|d| d.to_string()).unwrap_or_else(|| "– (runs on first use)".into());
        ui.label(RichText::new(format!("active: {active}")).weak());
        if let Some(f) = &app.cuda_fallback {
            ui.label(RichText::new("⚠ CUDA fallback").color(Color32::from_rgb(240, 170, 80)))
                .on_hover_text(f);
        }
    });
    ui.separator();

    let mut actions: Vec<(Action, &'static rembg_core::ModelSpec)> = Vec::new();
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        egui::Grid::new("models").striped(true).spacing([14.0, 8.0]).min_row_height(28.0).show(ui, |ui| {
            for h in ["Default", "Model", "Family", "Input", "Size", "License", "Status", ""] {
                ui.label(RichText::new(h).strong());
            }
            ui.end_row();
            for m in registry::MODELS {
                let is_installed = app.store.is_installed(m);
                let is_default = app.default_model == m.id;
                if ui.radio(is_default, "").on_hover_text("Set as default model").clicked() && !is_default {
                    actions.push((Action::Default, m));
                }
                ui.vertical(|ui| {
                    ui.label(RichText::new(m.name).strong());
                    ui.label(RichText::new(m.id).monospace().small().weak()).on_hover_text(m.description);
                });
                ui.label(m.family.label());
                ui.label(format!("{0}×{0}", m.pre.size));
                ui.label(human(m.size));
                let lic = RichText::new(m.license);
                ui.label(if m.commercial_ok { lic } else { lic.color(Color32::from_rgb(240, 170, 80)) })
                    .on_hover_text(if m.commercial_ok {
                        "Permissive licence"
                    } else {
                        "Non-commercial use only"
                    });
                if let Some((done, total)) = app.downloads.get(m.id) {
                    let frac = total.map(|t| *done as f32 / t as f32).unwrap_or(0.0);
                    ui.add(egui::ProgressBar::new(frac).desired_width(160.0).text(format!(
                        "{} / {}",
                        human(*done),
                        human(total.unwrap_or(m.size))
                    )));
                } else if app.worker.is_downloading(m.id) {
                    ui.spinner();
                } else if is_installed {
                    ui.label(RichText::new("✔ installed").color(Color32::from_rgb(110, 200, 120)));
                } else {
                    ui.label(RichText::new("not installed").weak());
                }
                ui.horizontal(|ui| {
                    if app.worker.is_downloading(m.id) {
                        if ui.button("Cancel").clicked() {
                            actions.push((Action::Cancel, m));
                        }
                    } else if is_installed {
                        if ui.button("🗑 Delete").clicked() {
                            actions.push((Action::Delete, m));
                        }
                        if ui.button("Verify").on_hover_text("Re-hash and compare sha256").clicked() {
                            actions.push((Action::Verify, m));
                        }
                    } else if ui.button("⬇ Download").clicked() {
                        actions.push((Action::Download, m));
                    }
                });
                ui.end_row();
            }
        });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("⬇ Download all").clicked() {
                for m in registry::MODELS {
                    actions.push((Action::Download, m));
                }
            }
        });
    });

    for (a, m) in actions {
        match a {
            Action::Default => app.set_default(m.id),
            Action::Download => app.worker.download(m),
            Action::Cancel => app.worker.cancel_download(m.id),
            Action::Delete => match app.store.remove(m) {
                Ok(()) => {
                    app.compare_models.remove(m.id);
                    app.set_status(format!("Deleted {}", m.id), false);
                }
                Err(e) => app.set_status(e.to_string(), true),
            },
            Action::Verify => match app.store.verify(m) {
                Ok(true) => app.set_status(format!("{}: checksum OK", m.id), false),
                Ok(false) => app.set_status(format!("{}: checksum MISMATCH — re-download it", m.id), true),
                Err(e) => app.set_status(e.to_string(), true),
            },
        }
    }
}

#[derive(Clone, Copy)]
enum Action {
    Default,
    Download,
    Cancel,
    Delete,
    Verify,
}
