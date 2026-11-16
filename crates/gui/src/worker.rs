//! Background inference + download threads. The UI thread never blocks on these.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, unbounded};
use image::{DynamicImage, GrayImage, RgbaImage};
use rembg_core::{ActiveDevice, DevicePref, Engine, ModelSpec, ModelStore, imageops};

/// Result of running one model on one image.
pub struct Output {
    pub mask: GrayImage,
    pub cutout: RgbaImage,
    pub infer: Duration,
    pub load: Option<Duration>,
    pub device: ActiveDevice,
}

enum Cmd {
    Run { job: u64, spec: &'static ModelSpec, image: Arc<DynamicImage> },
    SetDevice(DevicePref),
}

pub enum DlState {
    Progress(u64, Option<u64>),
    Finished,
    Failed(String),
}

pub enum Event {
    Started { job: u64, model: &'static str },
    Done { job: u64, model: &'static str, out: Arc<Output> },
    Failed { job: u64, model: &'static str, error: String },
    Device { device: ActiveDevice, fallback: Option<String> },
    Download { model: &'static str, state: DlState },
}

pub struct Worker {
    tx: Sender<Cmd>,
    pub rx: Receiver<Event>,
    ev_tx: Sender<Event>,
    /// Jobs below this id are stale and skipped.
    min_job: Arc<AtomicU64>,
    next_job: u64,
    store: ModelStore,
    ctx: egui::Context,
    downloads: HashMap<&'static str, Arc<AtomicBool>>,
}

impl Worker {
    pub fn spawn(ctx: egui::Context, store: ModelStore, pref: DevicePref) -> Self {
        let (tx, cmd_rx) = unbounded::<Cmd>();
        let (ev_tx, rx) = unbounded::<Event>();
        let min_job = Arc::new(AtomicU64::new(0));
        {
            let (ev_tx, min_job, store, ctx) = (ev_tx.clone(), min_job.clone(), store.clone(), ctx.clone());
            thread::Builder::new()
                .name("rembg-engine".into())
                .spawn(move || engine_loop(cmd_rx, ev_tx, min_job, store, ctx, pref))
                .expect("spawn engine thread");
        }
        Self { tx, rx, ev_tx, min_job, next_job: 1, store, ctx, downloads: HashMap::new() }
    }

    /// Start a new job id and cancel everything queued before it.
    pub fn new_job(&mut self) -> u64 {
        let id = self.next_job;
        self.next_job += 1;
        self.min_job.store(id, Ordering::Relaxed);
        id
    }

    pub fn run(&self, job: u64, spec: &'static ModelSpec, image: Arc<DynamicImage>) {
        let _ = self.tx.send(Cmd::Run { job, spec, image });
    }

    pub fn set_device(&self, pref: DevicePref) {
        let _ = self.tx.send(Cmd::SetDevice(pref));
    }

    pub fn is_downloading(&self, id: &str) -> bool {
        self.downloads.contains_key(id)
    }

    pub fn download_finished(&mut self, id: &str) {
        self.downloads.remove(id);
    }

    pub fn cancel_download(&mut self, id: &str) {
        if let Some(c) = self.downloads.get(id) {
            c.store(true, Ordering::Relaxed);
        }
    }

    pub fn download(&mut self, spec: &'static ModelSpec) {
        if self.downloads.contains_key(spec.id) || self.store.is_installed(spec) {
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.downloads.insert(spec.id, cancel.clone());
        let (store, tx, ctx) = (self.store.clone(), self.ev_tx.clone(), self.ctx.clone());
        thread::spawn(move || {
            let mut last = Instant::now();
            let res = store.download(spec, &cancel, |done, total| {
                if last.elapsed() > Duration::from_millis(100) {
                    last = Instant::now();
                    let _ =
                        tx.send(Event::Download { model: spec.id, state: DlState::Progress(done, total) });
                    ctx.request_repaint();
                }
            });
            let state = match res {
                Ok(_) => DlState::Finished,
                Err(e) => DlState::Failed(e.to_string()),
            };
            let _ = tx.send(Event::Download { model: spec.id, state });
            ctx.request_repaint();
        });
    }
}

fn engine_loop(
    rx: Receiver<Cmd>,
    tx: Sender<Event>,
    min_job: Arc<AtomicU64>,
    store: ModelStore,
    ctx: egui::Context,
    pref: DevicePref,
) {
    let mut engine = match Engine::new(pref) {
        Ok(e) => e,
        Err(e) => {
            tracing::error!("engine init failed: {e}");
            for cmd in rx {
                if let Cmd::Run { job, spec, .. } = cmd {
                    let _ = tx.send(Event::Failed { job, model: spec.id, error: e.to_string() });
                }
            }
            return;
        }
    };
    // Heavy BiRefNet sessions take ~1-4 GB each; keep two warm.
    engine.set_max_loaded(2);
    let send = |ev: Event| {
        let _ = tx.send(ev);
        ctx.request_repaint();
    };
    for cmd in rx {
        match cmd {
            Cmd::SetDevice(p) => engine.set_device_pref(p),
            Cmd::Run { job, spec, image } => {
                if job < min_job.load(Ordering::Relaxed) {
                    continue;
                }
                send(Event::Started { job, model: spec.id });
                let res = engine.predict(spec, &store.path(spec), &image);
                match res {
                    Ok(p) => {
                        send(Event::Device {
                            device: p.device,
                            fallback: engine.gpu_fallback_reason.clone(),
                        });
                        let cutout = imageops::cutout(&image, &p.mask);
                        let out = Output {
                            mask: p.mask,
                            cutout,
                            infer: p.infer_time,
                            load: p.load_time,
                            device: p.device,
                        };
                        send(Event::Done { job, model: spec.id, out: Arc::new(out) });
                    }
                    Err(e) => send(Event::Failed { job, model: spec.id, error: e.to_string() }),
                }
            }
        }
    }
}
