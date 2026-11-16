//! ONNX Runtime inference: image -> alpha mask.

use std::fmt;
use std::path::Path;
use std::time::{Duration, Instant};

use image::{DynamicImage, GrayImage};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::imageops;
use crate::registry::{ModelSpec, Scale};
use crate::{Error, Result};

/// Requested execution device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePref {
    /// CUDA if compiled in and available, else CPU.
    #[default]
    Auto,
    Cpu,
    Cuda,
}

/// Device a session actually runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveDevice {
    Cpu,
    Cuda,
}

impl fmt::Display for ActiveDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ActiveDevice::Cpu => "CPU",
            ActiveDevice::Cuda => "CUDA",
        })
    }
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub model: &'static str,
    /// Soft alpha mask at the source image's resolution.
    pub mask: GrayImage,
    pub device: ActiveDevice,
    /// Pre-process + inference + post-process time.
    pub infer_time: Duration,
    /// Session creation time, if the model had to be loaded for this call.
    pub load_time: Option<Duration>,
}

struct Loaded {
    id: &'static str,
    session: Session,
    device: ActiveDevice,
}

pub struct Engine {
    env: Environment,
    pref: DevicePref,
    /// LRU: most recently used at the end.
    loaded: Vec<Loaded>,
    max_loaded: usize,
    /// Why CUDA was not used (if it was requested/auto and failed).
    pub cuda_fallback_reason: Option<String>,
}

fn ort_err(e: impl fmt::Display) -> Error {
    Error::Ort(e.to_string())
}

impl Engine {
    pub fn new(pref: DevicePref) -> Result<Self> {
        let env = ort::environment::init().with_name("rembg").build().map_err(ort_err)?;
        Ok(Self { env, pref, loaded: Vec::new(), max_loaded: 2, cuda_fallback_reason: None })
    }

    /// Whether this binary was built with the CUDA execution provider.
    pub const fn cuda_compiled() -> bool {
        cfg!(feature = "cuda")
    }

    pub fn device_pref(&self) -> DevicePref {
        self.pref
    }

    /// Change the device preference; drops loaded sessions.
    pub fn set_device_pref(&mut self, pref: DevicePref) {
        if pref != self.pref {
            self.pref = pref;
            self.unload_all();
            self.cuda_fallback_reason = None;
        }
    }

    /// Max number of sessions kept in memory (heavy models use GBs).
    pub fn set_max_loaded(&mut self, n: usize) {
        self.max_loaded = n.max(1);
        self.evict();
    }

    pub fn unload_all(&mut self) {
        self.loaded.clear();
    }

    pub fn is_loaded(&self, id: &str) -> bool {
        self.loaded.iter().any(|l| l.id == id)
    }

    fn evict(&mut self) {
        while self.loaded.len() > self.max_loaded {
            self.loaded.remove(0);
        }
    }

    fn build_session(&self, path: &Path, cuda: bool) -> Result<Session> {
        #[allow(unused_mut)]
        let mut b = Session::builder(&self.env)
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .unwrap_or_else(|e| e.recover());
        if cuda {
            #[cfg(feature = "cuda")]
            {
                b = b
                    .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                    .map_err(ort_err)?;
            }
            #[cfg(not(feature = "cuda"))]
            return Err(Error::Other("built without CUDA support (rebuild with --features cuda)".into()));
        }
        b.commit_from_file(path).map_err(ort_err)
    }

    /// Load (or fetch from cache) the session for `spec`. Returns its index and load time.
    fn ensure_loaded(&mut self, spec: &'static ModelSpec, path: &Path) -> Result<Option<Duration>> {
        if let Some(i) = self.loaded.iter().position(|l| l.id == spec.id) {
            let l = self.loaded.remove(i);
            self.loaded.push(l);
            return Ok(None);
        }
        if !path.is_file() {
            return Err(Error::NotInstalled(spec.id.into()));
        }
        // Free memory before loading another big model.
        self.max_loaded = self.max_loaded.max(1);
        while self.loaded.len() >= self.max_loaded {
            self.loaded.remove(0);
        }
        let t = Instant::now();
        let want_cuda = match self.pref {
            DevicePref::Cpu => false,
            DevicePref::Cuda => true,
            DevicePref::Auto => Self::cuda_compiled() && self.cuda_fallback_reason.is_none(),
        };
        let (session, device) = if want_cuda {
            match self.build_session(path, true) {
                Ok(s) => (s, ActiveDevice::Cuda),
                Err(e) if self.pref == DevicePref::Auto => {
                    tracing::warn!("CUDA unavailable, falling back to CPU: {e}");
                    self.cuda_fallback_reason = Some(e.to_string());
                    (self.build_session(path, false)?, ActiveDevice::Cpu)
                }
                Err(e) => return Err(e),
            }
        } else {
            (self.build_session(path, false)?, ActiveDevice::Cpu)
        };
        self.loaded.push(Loaded { id: spec.id, session, device });
        Ok(Some(t.elapsed()))
    }

    /// Run `spec` on `img`, returning a soft mask at the image's resolution.
    pub fn predict(
        &mut self,
        spec: &'static ModelSpec,
        model_path: &Path,
        img: &DynamicImage,
    ) -> Result<Prediction> {
        let load_time = self.ensure_loaded(spec, model_path)?;
        let t = Instant::now();
        let input = preprocess(spec, img)?;
        let s = spec.pre.size as usize;
        let tensor = Tensor::from_array(([1usize, 3, s, s], input)).map_err(ort_err)?;

        let loaded = self.loaded.last_mut().expect("just loaded");
        let device = loaded.device;
        let outputs = loaded.session.run(ort::inputs![tensor]).map_err(ort_err)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        if dims.len() < 2 {
            return Err(Error::Other(format!("unexpected output shape {dims:?}")));
        }
        let (oh, ow) = (dims[dims.len() - 2] as u32, dims[dims.len() - 1] as u32);
        let plane = &data[..(oh * ow) as usize];
        let small = postprocess(spec, plane, ow, oh);
        drop(outputs);

        let mask = imageops::resize_gray(&small, img.width(), img.height())?;
        Ok(Prediction { model: spec.id, mask, device, infer_time: t.elapsed(), load_time })
    }
}

/// RGB -> normalized NCHW f32 tensor data.
fn preprocess(spec: &ModelSpec, img: &DynamicImage) -> Result<Vec<f32>> {
    let p = spec.pre;
    let rgb = imageops::resize_rgb(&img.to_rgb8(), p.size, p.size)?;
    let scale = match p.scale {
        Scale::MaxPixel => rgb.as_raw().iter().copied().max().unwrap_or(0).max(1) as f32,
        Scale::Div255 => 255.0,
    };
    let n = (p.size * p.size) as usize;
    let mut out = vec![0f32; 3 * n];
    for (i, px) in rgb.pixels().enumerate() {
        for c in 0..3 {
            out[c * n + i] = (px[c] as f32 / scale - p.mean[c]) / p.std[c];
        }
    }
    Ok(out)
}

/// Raw model output plane -> 8-bit mask (sigmoid optional, then min-max normalize).
fn postprocess(spec: &ModelSpec, plane: &[f32], w: u32, h: u32) -> GrayImage {
    let vals: Vec<f32> = if spec.post.sigmoid {
        plane.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
    } else {
        plane.to_vec()
    };
    let (mi, ma) = vals.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let range = (ma - mi).max(1e-6);
    let bytes = vals.iter().map(|v| (((v - mi) / range) * 255.0).clamp(0.0, 255.0) as u8).collect();
    GrayImage::from_raw(w, h, bytes).expect("dims match")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry;

    #[test]
    fn postprocess_normalizes() {
        let spec = registry::find("u2netp").unwrap();
        let m = postprocess(spec, &[0.2, 0.4, 0.6, 1.0], 2, 2);
        assert_eq!(m.as_raw()[0], 0);
        assert_eq!(m.as_raw()[3], 255);
    }

    #[test]
    fn preprocess_shape() {
        let spec = registry::find("u2netp").unwrap();
        let img = DynamicImage::new_rgb8(40, 30);
        assert_eq!(preprocess(spec, &img).unwrap().len(), 3 * 320 * 320);
    }
}
