//! Static catalogue of supported background-removal models.
//!
//! Pre/post-processing parameters mirror the reference Python `rembg` sessions
//! so outputs are comparable.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    U2Net,
    IsNet,
    BiRefNet,
    Bria,
}

impl Family {
    pub fn label(self) -> &'static str {
        match self {
            Family::U2Net => "U²-Net",
            Family::IsNet => "ISNet",
            Family::BiRefNet => "BiRefNet",
            Family::Bria => "BRIA RMBG",
        }
    }
}

/// How raw `u8` pixels are scaled to `[0, 1]` before mean/std normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    /// Divide by the image's max channel value (rembg's behaviour).
    MaxPixel,
    /// Divide by 255.
    Div255,
}

#[derive(Debug, Clone, Copy)]
pub struct Preprocess {
    pub size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub scale: Scale,
}

#[derive(Debug, Clone, Copy)]
pub struct Postprocess {
    /// Apply a sigmoid to the raw output before min-max normalization.
    pub sigmoid: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub family: Family,
    pub description: &'static str,
    pub url: &'static str,
    /// Download size in bytes.
    pub size: u64,
    pub sha256: &'static str,
    pub license: &'static str,
    pub commercial_ok: bool,
    pub pre: Preprocess,
    pub post: Postprocess,
}

impl ModelSpec {
    pub fn file_name(&self) -> String {
        format!("{}.onnx", self.id)
    }
}

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const ONES: [f32; 3] = [1.0, 1.0, 1.0];

const fn pre(size: u32, mean: [f32; 3], std: [f32; 3]) -> Preprocess {
    Preprocess { size, mean, std, scale: Scale::MaxPixel }
}
const U2: Preprocess = pre(320, IMAGENET_MEAN, IMAGENET_STD);
const BIREF: Preprocess = pre(1024, IMAGENET_MEAN, IMAGENET_STD);
const RAW: Postprocess = Postprocess { sigmoid: false };
const SIG: Postprocess = Postprocess { sigmoid: true };

macro_rules! url {
    ($f:literal) => {
        concat!("https://github.com/danielgatis/rembg/releases/download/v0.0.0/", $f)
    };
}

pub static MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "u2net",
        name: "U²-Net",
        family: Family::U2Net,
        description: "General-purpose salient object model (2020). Fast, soft edges.",
        url: url!("u2net.onnx"),
        size: 175_997_641,
        sha256: "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
        license: "Apache-2.0",
        commercial_ok: true,
        pre: U2,
        post: RAW,
    },
    ModelSpec {
        id: "u2netp",
        name: "U²-Net (small)",
        family: Family::U2Net,
        description: "Tiny U²-Net (4.6 MB). Very fast, lower quality.",
        url: url!("u2netp.onnx"),
        size: 4_574_861,
        sha256: "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        license: "Apache-2.0",
        commercial_ok: true,
        pre: U2,
        post: RAW,
    },
    ModelSpec {
        id: "u2net_human_seg",
        name: "U²-Net human",
        family: Family::U2Net,
        description: "U²-Net trained for human segmentation.",
        url: url!("u2net_human_seg.onnx"),
        size: 175_997_641,
        sha256: "01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c",
        license: "Apache-2.0",
        commercial_ok: true,
        pre: U2,
        post: RAW,
    },
    ModelSpec {
        id: "silueta",
        name: "Silueta",
        family: Family::U2Net,
        description: "Pruned U²-Net (43 MB), similar quality to u2net.",
        url: url!("silueta.onnx"),
        size: 44_173_029,
        sha256: "75da6c8d2f8096ec743d071951be73b4a8bc7b3e51d9a6625d63644f90ffeedb",
        license: "Apache-2.0",
        commercial_ok: true,
        pre: U2,
        post: RAW,
    },
    ModelSpec {
        id: "isnet-general-use",
        name: "ISNet general",
        family: Family::IsNet,
        description: "DIS / ISNet general use (2022). Sharp edges, good default.",
        url: url!("isnet-general-use.onnx"),
        size: 178_648_008,
        sha256: "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        license: "Apache-2.0",
        commercial_ok: true,
        pre: pre(1024, [0.5, 0.5, 0.5], ONES),
        post: RAW,
    },
    ModelSpec {
        id: "isnet-anime",
        name: "ISNet anime",
        family: Family::IsNet,
        description: "ISNet fine-tuned for anime / illustration characters.",
        url: url!("isnet-anime.onnx"),
        size: 176_069_933,
        sha256: "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99",
        license: "Apache-2.0",
        commercial_ok: true,
        pre: pre(1024, IMAGENET_MEAN, ONES),
        post: RAW,
    },
    ModelSpec {
        id: "birefnet-general",
        name: "BiRefNet general",
        family: Family::BiRefNet,
        description: "BiRefNet (2024), Swin-L backbone. State-of-the-art general quality; heavy.",
        url: url!("BiRefNet-general-epoch_244.onnx"),
        size: 972_666_916,
        sha256: "58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c",
        license: "MIT",
        commercial_ok: true,
        pre: BIREF,
        post: SIG,
    },
    ModelSpec {
        id: "birefnet-general-lite",
        name: "BiRefNet lite",
        family: Family::BiRefNet,
        description: "BiRefNet with Swin-T backbone. Much faster, still very good.",
        url: url!("BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"),
        size: 224_005_088,
        sha256: "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
        license: "MIT",
        commercial_ok: true,
        pre: BIREF,
        post: SIG,
    },
    ModelSpec {
        id: "birefnet-portrait",
        name: "BiRefNet portrait",
        family: Family::BiRefNet,
        description: "BiRefNet trained for portraits / people (hair detail).",
        url: url!("BiRefNet-portrait-epoch_150.onnx"),
        size: 972_666_916,
        sha256: "1ba1c8ff5a7bbfadc8d8d13fb11d7be793f91f23d9d466549e37a854f6668f99",
        license: "MIT",
        commercial_ok: true,
        pre: BIREF,
        post: SIG,
    },
    ModelSpec {
        id: "birefnet-dis",
        name: "BiRefNet DIS",
        family: Family::BiRefNet,
        description: "BiRefNet trained on DIS5K: dichotomous segmentation, fine structures.",
        url: url!("BiRefNet-DIS-epoch_590.onnx"),
        size: 972_666_916,
        sha256: "6470117bac6f8d82a3f62921056f52d0f5c4d36d1d832096331d5ea38a03acb5",
        license: "MIT",
        commercial_ok: true,
        pre: BIREF,
        post: SIG,
    },
    ModelSpec {
        id: "birefnet-hrsod",
        name: "BiRefNet HRSOD",
        family: Family::BiRefNet,
        description: "BiRefNet for high-resolution salient object detection.",
        url: url!("BiRefNet-HRSOD_DHU-epoch_115.onnx"),
        size: 972_666_916,
        sha256: "4f5837663194fb88f603b76782eae05a3c29f5749872ca1bfb636bd26e7f6bfc",
        license: "MIT",
        commercial_ok: true,
        pre: BIREF,
        post: SIG,
    },
    ModelSpec {
        id: "birefnet-massive",
        name: "BiRefNet massive",
        family: Family::BiRefNet,
        description: "BiRefNet trained on a massive combined dataset. Most robust; heavy.",
        url: url!("BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx"),
        size: 972_666_916,
        sha256: "a94814cac438a31f95287811882628644a04b22d313ef3071d2ba904b5f627b8",
        license: "MIT",
        commercial_ok: true,
        pre: BIREF,
        post: SIG,
    },
    ModelSpec {
        id: "bria-rmbg-1.4",
        name: "BRIA RMBG 1.4",
        family: Family::Bria,
        description: "BRIA RMBG v1.4 (IS-Net based), trained on licensed data.",
        url: "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx",
        size: 176_153_355,
        sha256: "8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f01535ba742e0f",
        license: "BRIA RMBG-1.4 (non-commercial)",
        commercial_ok: false,
        pre: Preprocess { size: 1024, mean: [0.5, 0.5, 0.5], std: ONES, scale: Scale::Div255 },
        post: RAW,
    },
    ModelSpec {
        id: "bria-rmbg-2.0",
        name: "BRIA RMBG 2.0",
        family: Family::Bria,
        description: "BRIA RMBG v2.0 (BiRefNet based). Excellent quality; heavy.",
        url: url!("bria-rmbg-2.0.onnx"),
        size: 1_024_331_469,
        sha256: "5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958",
        license: "CC BY-NC 4.0 (non-commercial)",
        commercial_ok: false,
        pre: BIREF,
        post: RAW,
    },
];

/// Model used when none is specified and no default is configured.
pub const FALLBACK_DEFAULT: &str = "isnet-general-use";

pub fn find(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_unique_and_fallback_exists() {
        let mut ids: Vec<_> = MODELS.iter().map(|m| m.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), MODELS.len());
        assert!(find(FALLBACK_DEFAULT).is_some());
    }

    #[test]
    fn hashes_pinned() {
        for m in MODELS {
            assert_eq!(m.sha256.len(), 64, "{} sha256 missing", m.id);
            assert!(m.size > 0, "{} size missing", m.id);
        }
    }
}
