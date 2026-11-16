//! `rembg-gui`: launches the GUI directly. On Windows this is a GUI-subsystem
//! executable, so double-clicking it doesn't open a console window.
//!
//! Usage: `rembg-gui [image]` (also works as a file-association / "Open with" target).
#![cfg_attr(windows, windows_subsystem = "windows")]

fn main() {
    let image = std::env::args_os().nth(1).map(std::path::PathBuf::from);
    if let Err(e) = rembg_gui::run(rembg_gui::Launch { image, ..Default::default() }) {
        // There may be no console to print to: show the error in a dialog too.
        eprintln!("rembg-gui: {e:#}");
        rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Error)
            .set_title("rembg")
            .set_description(format!("{e:#}"))
            .show();
        std::process::exit(1);
    }
}
