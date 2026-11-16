//! On-disk model cache (`~/.cache/rembg/models`) and user config.

use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::registry::{self, ModelSpec};
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct ModelStore {
    root: PathBuf,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    pub default_model: Option<String>,
}

impl ModelStore {
    /// `$REMBG_HOME`, else `~/.cache/rembg` (Linux/macOS) or `%LOCALAPPDATA%\rembg` (Windows).
    pub fn open_default() -> Result<Self> {
        let default = if cfg!(windows) {
            dirs::data_local_dir().map(|d| d.join("rembg"))
        } else {
            dirs::home_dir().map(|h| h.join(".cache").join("rembg"))
        };
        let root = std::env::var_os("REMBG_HOME")
            .map(PathBuf::from)
            .or(default)
            .ok_or_else(|| Error::Other("cannot determine home directory".into()))?;
        Self::open(root)
    }

    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("models"))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    pub fn path(&self, spec: &ModelSpec) -> PathBuf {
        self.models_dir().join(spec.file_name())
    }

    pub fn is_installed(&self, spec: &ModelSpec) -> bool {
        self.path(spec).is_file()
    }

    pub fn installed(&self) -> Vec<&'static ModelSpec> {
        registry::MODELS.iter().filter(|m| self.is_installed(m)).collect()
    }

    /// Size on disk if installed.
    pub fn disk_size(&self, spec: &ModelSpec) -> Option<u64> {
        fs::metadata(self.path(spec)).ok().map(|m| m.len())
    }

    pub fn remove(&self, spec: &ModelSpec) -> Result<()> {
        let p = self.path(spec);
        if p.exists() {
            fs::remove_file(p)?;
        }
        Ok(())
    }

    /// Download `spec` into the cache, verifying its sha256.
    ///
    /// `progress(done, total)` is called periodically; setting `cancel` aborts.
    pub fn download(
        &self,
        spec: &ModelSpec,
        cancel: &AtomicBool,
        mut progress: impl FnMut(u64, Option<u64>),
    ) -> Result<PathBuf> {
        let dest = self.path(spec);
        let part = dest.with_extension("onnx.part");
        let mut resp = ureq::get(spec.url).call().map_err(|e| Error::Download(e.to_string()))?;
        let total = resp
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
            .or(Some(spec.size).filter(|s| *s > 0));
        let mut reader = resp.body_mut().as_reader();
        let mut out = File::create(&part)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        let mut done = 0u64;
        let result = loop {
            if cancel.load(Ordering::Relaxed) {
                break Err(Error::Cancelled);
            }
            let n = match reader.read(&mut buf) {
                Ok(0) => break Ok(()),
                Ok(n) => n,
                Err(e) => break Err(Error::Download(e.to_string())),
            };
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n])?;
            done += n as u64;
            progress(done, total);
        };
        drop(out);
        if let Err(e) = result {
            let _ = fs::remove_file(&part);
            return Err(e);
        }
        let got = hex(&hasher.finalize());
        if !spec.sha256.is_empty() && got != spec.sha256 {
            let _ = fs::remove_file(&part);
            return Err(Error::Checksum { model: spec.id.into(), expected: spec.sha256.into(), got });
        }
        fs::rename(&part, &dest)?;
        Ok(dest)
    }

    /// Re-hash an installed model and compare with the registry.
    pub fn verify(&self, spec: &ModelSpec) -> Result<bool> {
        let mut f = BufReader::new(File::open(self.path(spec))?);
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        Ok(hex(&h.finalize()) == spec.sha256)
    }

    fn config_path(&self) -> PathBuf {
        self.root.join("config.json")
    }

    pub fn config(&self) -> Config {
        fs::read(self.config_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    pub fn save_config(&self, cfg: &Config) -> Result<()> {
        fs::write(self.config_path(), serde_json::to_vec_pretty(cfg).expect("serializable"))?;
        Ok(())
    }

    /// Configured default model, falling back to the registry default.
    pub fn default_model(&self) -> &'static ModelSpec {
        self.config()
            .default_model
            .as_deref()
            .and_then(registry::find)
            .unwrap_or_else(|| registry::find(registry::FALLBACK_DEFAULT).expect("fallback exists"))
    }

    pub fn set_default_model(&self, id: &str) -> Result<()> {
        registry::find(id).ok_or_else(|| Error::UnknownModel(id.into()))?;
        let mut cfg = self.config();
        cfg.default_model = Some(id.into());
        self.save_config(&cfg)
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_roundtrip() {
        let dir = std::env::temp_dir().join(format!("rembg-test-{}", std::process::id()));
        let store = ModelStore::open(&dir).unwrap();
        assert_eq!(store.default_model().id, registry::FALLBACK_DEFAULT);
        store.set_default_model("u2netp").unwrap();
        assert_eq!(store.default_model().id, "u2netp");
        assert!(store.set_default_model("nope").is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
