use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use image::{DynamicImage, ImageFormat, RgbaImage};
use indicatif::{ProgressBar, ProgressStyle};
use rembg_core::grid::{self, Tile, TileBg};
use rembg_core::{DevicePref, Engine, ModelSpec, ModelStore, imageops, registry};

#[derive(Parser)]
#[command(name = "rembg", version, about = "Native local background removal")]
struct Cli {
    /// Execution device
    #[arg(long, global = true, value_enum, default_value_t = Device::Auto)]
    device: Device,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Device {
    Auto,
    Cpu,
    Cuda,
}

impl From<Device> for DevicePref {
    fn from(d: Device) -> Self {
        match d {
            Device::Auto => DevicePref::Auto,
            Device::Cpu => DevicePref::Cpu,
            Device::Cuda => DevicePref::Cuda,
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Remove the background from images
    Remove(RemoveArgs),
    /// Run an image through several models and save a labelled grid
    Compare(CompareArgs),
    /// Manage models
    #[command(subcommand)]
    Models(ModelsCmd),
    /// Launch the GUI
    Gui {
        /// Image to open
        image: Option<PathBuf>,
    },
    /// Show runtime / device info
    Info,
    #[command(hide = true)]
    ServeClipboard { png: PathBuf },
}

#[derive(Args)]
struct RemoveArgs {
    /// Input files or directories ("-" = stdin)
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Output file (single input) or directory ("-" = stdout). Default: <name>_nobg.<ext> next to input
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Model id (see `rembg models list`)
    #[arg(short, long)]
    model: Option<String>,
    /// Output only the alpha mask
    #[arg(long)]
    mask: bool,
    /// Flatten onto a background colour (#rrggbb or name) instead of transparency
    #[arg(long)]
    bg: Option<String>,
    /// Output format when writing to a directory/stdout
    #[arg(short, long, value_enum, default_value_t = Fmt::Png)]
    format: Fmt,
    /// Hard-threshold the mask at this value (0-255)
    #[arg(long)]
    threshold: Option<u8>,
    /// Copy the (single) result to the clipboard
    #[arg(long)]
    copy: bool,
}

#[derive(Args)]
struct CompareArgs {
    image: PathBuf,
    /// Output grid image
    #[arg(short, long, default_value = "compare.png")]
    output: PathBuf,
    /// Comma-separated model ids (default: all installed)
    #[arg(short, long, value_delimiter = ',')]
    models: Vec<String>,
    /// Tile size in px
    #[arg(long, default_value_t = 512)]
    tile: u32,
    /// Tile background: "checker" or a colour
    #[arg(long, default_value = "checker")]
    bg: String,
    /// Show masks instead of cut-outs
    #[arg(long)]
    mask: bool,
    /// Also save each individual result into this directory
    #[arg(long)]
    save_each: Option<PathBuf>,
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// List models and install state
    List,
    /// Download models
    Pull {
        ids: Vec<String>,
        #[arg(long)]
        all: bool,
    },
    /// Delete downloaded models
    Rm { ids: Vec<String> },
    /// Re-hash installed models
    Verify,
    /// Show or set the default model
    Default { id: Option<String> },
    /// Print the cache directory
    Path,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum Fmt {
    Png,
    Webp,
    Jpg,
}

impl Fmt {
    fn ext(self) -> &'static str {
        match self {
            Fmt::Png => "png",
            Fmt::Webp => "webp",
            Fmt::Jpg => "jpg",
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn,ort=error".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let store = ModelStore::open_default()?;
    match cli.cmd {
        Cmd::Remove(a) => remove(&store, cli.device.into(), a),
        Cmd::Compare(a) => compare(&store, cli.device.into(), a),
        Cmd::Models(m) => models(&store, m),
        Cmd::Gui { image } => rembg_gui::run(image, cli.device.into()),
        Cmd::Info => info(&store, cli.device.into()),
        Cmd::ServeClipboard { png } => serve_clipboard(&png),
    }
}

fn spec(id: &str) -> Result<&'static ModelSpec> {
    registry::find(id).with_context(|| format!("unknown model '{id}' (see `rembg models list`)"))
}

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else {
        format!("{:.0} KB", b / 1e3)
    }
}

/// Download with a progress bar if not installed.
fn ensure_installed(store: &ModelStore, spec: &'static ModelSpec) -> Result<PathBuf> {
    if store.is_installed(spec) {
        return Ok(store.path(spec));
    }
    if !spec.commercial_ok {
        eprintln!("note: {} is licensed {}", spec.id, spec.license);
    }
    let pb = ProgressBar::new(spec.size);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg:24} [{bar:30}] {bytes}/{total_bytes} {bytes_per_sec} eta {eta}",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_message(format!("downloading {}", spec.id));
    let path = store.download(spec, &AtomicBool::new(false), |done, total| {
        if let Some(t) = total {
            pb.set_length(t);
        }
        pb.set_position(done);
    })?;
    pb.finish_and_clear();
    eprintln!("installed {} ({})", spec.id, human(spec.size));
    Ok(path)
}

fn collect_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    const EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"];
    let mut out = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut v: Vec<_> = std::fs::read_dir(p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|f| {
                    f.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| EXTS.contains(&e.to_ascii_lowercase().as_str()))
                })
                .collect();
            v.sort();
            out.extend(v);
        } else {
            out.push(p.clone());
        }
    }
    Ok(out)
}

fn render(img: &DynamicImage, mask: &image::GrayImage, a: &RemoveArgs, bg: Option<[u8; 3]>) -> DynamicImage {
    if a.mask {
        return DynamicImage::ImageLuma8(mask.clone());
    }
    let cut = imageops::cutout(img, mask);
    match bg {
        Some(c) => DynamicImage::ImageRgb8(imageops::flatten(&cut, c)),
        None if a.format == Fmt::Jpg => DynamicImage::ImageRgb8(imageops::flatten(&cut, [255; 3])),
        None => DynamicImage::ImageRgba8(cut),
    }
}

fn remove(store: &ModelStore, dev: DevicePref, a: RemoveArgs) -> Result<()> {
    let spec = match &a.model {
        Some(id) => spec(id)?,
        None => store.default_model(),
    };
    let bg = a.bg.as_deref().map(|s| imageops::parse_color(s).with_context(|| format!("bad colour '{s}'"))).transpose()?;
    let inputs = collect_inputs(&a.inputs)?;
    if inputs.is_empty() {
        bail!("no input images");
    }
    if a.copy && inputs.len() != 1 {
        bail!("--copy needs exactly one input");
    }
    let model_path = ensure_installed(store, spec)?;
    let mut engine = Engine::new(dev)?;
    let single_file_out = inputs.len() == 1
        && a.output.as_ref().is_some_and(|o| o.extension().is_some() && !o.is_dir());
    if let Some(o) = &a.output
        && !single_file_out
        && o.as_os_str() != "-"
    {
        std::fs::create_dir_all(o)?;
    }

    for input in &inputs {
        let img = if input.as_os_str() == "-" {
            let mut buf = Vec::new();
            std::io::stdin().read_to_end(&mut buf)?;
            imageops::load_from_memory(&buf)?
        } else {
            imageops::load(input).with_context(|| format!("loading {}", input.display()))?
        };
        let mut pred = engine.predict(spec, &model_path, &img)?;
        if let Some(t) = a.threshold {
            imageops::threshold(&mut pred.mask, t);
        }
        let out_img = render(&img, &pred.mask, &a, bg);

        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("stdin");
        let dest: Option<PathBuf> = match &a.output {
            Some(o) if o.as_os_str() == "-" => None,
            Some(o) if single_file_out => Some(o.clone()),
            Some(o) => Some(o.join(format!("{stem}.{}", a.format.ext()))),
            None if input.as_os_str() == "-" => None,
            None => Some(input.with_file_name(format!("{stem}_nobg.{}", a.format.ext()))),
        };
        match &dest {
            Some(d) => {
                save(&out_img, d)?;
                eprintln!(
                    "{} -> {}  [{} on {}, {:.0} ms{}]",
                    input.display(),
                    d.display(),
                    spec.id,
                    pred.device,
                    pred.infer_time.as_secs_f64() * 1e3,
                    pred.load_time.map(|l| format!(", load {:.1}s", l.as_secs_f64())).unwrap_or_default()
                );
            }
            None => {
                let fmt = ImageFormat::from_extension(a.format.ext()).unwrap();
                let mut buf = std::io::Cursor::new(Vec::new());
                out_img.write_to(&mut buf, fmt)?;
                std::io::stdout().write_all(buf.get_ref())?;
            }
        }
        if a.copy {
            copy_to_clipboard(&out_img.to_rgba8())?;
        }
    }
    Ok(())
}

fn save(img: &DynamicImage, path: &Path) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let is_jpg = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg"));
    if is_jpg && img.color().has_alpha() {
        DynamicImage::ImageRgb8(imageops::flatten(&img.to_rgba8(), [255; 3])).save(path)?;
    } else {
        img.save(path)?;
    }
    Ok(())
}

/// Clipboard contents on X11/Wayland live in the owning process, so hand the
/// image to a detached child that serves it until something else is copied.
fn copy_to_clipboard(img: &RgbaImage) -> Result<()> {
    let tmp = std::env::temp_dir().join(format!("rembg-clip-{}.png", std::process::id()));
    img.save(&tmp)?;
    std::process::Command::new(std::env::current_exe()?)
        .arg("serve-clipboard")
        .arg(&tmp)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("spawning clipboard server")?;
    eprintln!("copied {}x{} image to clipboard", img.width(), img.height());
    Ok(())
}

fn serve_clipboard(png: &Path) -> Result<()> {
    let img = image::open(png)?.to_rgba8();
    let _ = std::fs::remove_file(png);
    let data = arboard::ImageData {
        width: img.width() as usize,
        height: img.height() as usize,
        bytes: img.into_raw().into(),
    };
    let mut cb = arboard::Clipboard::new()?;
    #[cfg(target_os = "linux")]
    {
        use arboard::SetExtLinux;
        cb.set().wait().image(data)?;
    }
    #[cfg(not(target_os = "linux"))]
    cb.set_image(data)?;
    Ok(())
}

fn compare(store: &ModelStore, dev: DevicePref, a: CompareArgs) -> Result<()> {
    let specs: Vec<&'static ModelSpec> = if a.models.is_empty() {
        store.installed()
    } else {
        a.models.iter().map(|id| spec(id)).collect::<Result<_>>()?
    };
    if specs.is_empty() {
        bail!("no models installed (rembg models pull <id>)");
    }
    let bg = match a.bg.as_str() {
        "checker" => TileBg::Checker,
        c => TileBg::Solid(imageops::parse_color(c).with_context(|| format!("bad colour '{c}'"))?),
    };
    let img = imageops::load(&a.image)?;
    let mut engine = Engine::new(dev)?;
    engine.set_max_loaded(1);
    if let Some(d) = &a.save_each {
        std::fs::create_dir_all(d)?;
    }
    let mut results: Vec<(String, RgbaImage)> = Vec::new();
    for s in &specs {
        let path = ensure_installed(store, s)?;
        let p = engine.predict(s, &path, &img)?;
        let rgba = if a.mask { imageops::mask_rgba(&p.mask) } else { imageops::cutout(&img, &p.mask) };
        eprintln!(
            "{:24} {:>7.0} ms  (load {:.1}s, {})",
            s.id,
            p.infer_time.as_secs_f64() * 1e3,
            p.load_time.unwrap_or_default().as_secs_f64(),
            p.device
        );
        if let Some(d) = &a.save_each {
            rgba.save(d.join(format!("{}.png", s.id)))?;
        }
        results.push((format!("{} · {:.0} ms", s.name, p.infer_time.as_secs_f64() * 1e3), rgba));
    }
    let tiles: Vec<Tile> = results.iter().map(|(l, i)| Tile { label: l.clone(), image: i }).collect();
    grid::compose(&tiles, a.tile, bg)?.save(&a.output)?;
    eprintln!("wrote {}", a.output.display());
    Ok(())
}

fn models(store: &ModelStore, cmd: ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::List => {
            let def = store.default_model().id;
            println!("{:2} {:24} {:10} {:>9}  {:5} LICENSE", "", "ID", "FAMILY", "SIZE", "INPUT");
            for m in registry::MODELS {
                let mark = match (store.is_installed(m), m.id == def) {
                    (true, true) => "*✓",
                    (true, false) => " ✓",
                    (false, true) => "* ",
                    _ => "  ",
                };
                println!(
                    "{mark} {:24} {:10} {:>9}  {:5} {}",
                    m.id,
                    m.family.label(),
                    human(m.size),
                    m.pre.size,
                    m.license
                );
            }
            let used: u64 = store.installed().iter().filter_map(|m| store.disk_size(m)).sum();
            println!("\n✓ installed  * default    cache: {} ({})", store.models_dir().display(), human(used));
        }
        ModelsCmd::Pull { ids, all } => {
            let list: Vec<&'static ModelSpec> = if all {
                registry::MODELS.iter().collect()
            } else if ids.is_empty() {
                vec![store.default_model()]
            } else {
                ids.iter().map(|i| spec(i)).collect::<Result<_>>()?
            };
            for s in list {
                if store.is_installed(s) {
                    eprintln!("{} already installed", s.id);
                } else {
                    ensure_installed(store, s)?;
                }
            }
        }
        ModelsCmd::Rm { ids } => {
            for i in ids {
                store.remove(spec(&i)?)?;
                eprintln!("removed {i}");
            }
        }
        ModelsCmd::Verify => {
            for s in store.installed() {
                let ok = store.verify(s)?;
                println!("{:24} {}", s.id, if ok { "ok" } else { "CORRUPT" });
            }
        }
        ModelsCmd::Default { id } => match id {
            Some(i) => {
                store.set_default_model(&i)?;
                println!("default model: {i}");
            }
            None => println!("{}", store.default_model().id),
        },
        ModelsCmd::Path => println!("{}", store.models_dir().display()),
    }
    Ok(())
}

fn info(store: &ModelStore, dev: DevicePref) -> Result<()> {
    println!("rembg {}", env!("CARGO_PKG_VERSION"));
    println!("cuda compiled: {}", Engine::cuda_compiled());
    println!("cache: {}", store.root().display());
    println!("default model: {}", store.default_model().id);
    let def = store.default_model();
    if let Some(s) = store.installed().into_iter().find(|m| m.id == def.id).or_else(|| store.installed().first().copied()) {
        let mut e = Engine::new(dev)?;
        let img = DynamicImage::new_rgb8(64, 64);
        let p = e.predict(s, &store.path(s), &img)?;
        println!("probe with {}: running on {}", s.id, p.device);
        if let Some(r) = &e.cuda_fallback_reason {
            println!("cuda fallback reason: {r}");
        }
    }
    Ok(())
}
