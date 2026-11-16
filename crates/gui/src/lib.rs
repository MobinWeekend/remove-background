//! Native GUI.
use std::path::PathBuf;

pub fn run(_image: Option<PathBuf>, _device: rembg_core::DevicePref) -> anyhow::Result<()> {
    anyhow::bail!("GUI not implemented yet")
}
