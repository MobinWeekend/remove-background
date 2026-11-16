//! Image loading, resizing and compositing helpers.

use std::path::Path;

use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{
    DynamicImage, GrayImage, ImageBuffer, ImageDecoder, ImageReader, Luma, Pixel, Rgb, RgbImage,
    Rgba, RgbaImage,
};

use crate::{Error, Result};

/// Load an image from disk, honouring EXIF orientation.
pub fn load(path: impl AsRef<Path>) -> Result<DynamicImage> {
    let mut dec = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    let orientation = dec.orientation()?;
    let mut img = DynamicImage::from_decoder(dec)?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// Decode an image from memory, honouring EXIF orientation.
pub fn load_from_memory(bytes: &[u8]) -> Result<DynamicImage> {
    let mut dec =
        ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?.into_decoder()?;
    let orientation = dec.orientation()?;
    let mut img = DynamicImage::from_decoder(dec)?;
    img.apply_orientation(orientation);
    Ok(img)
}

fn lanczos() -> ResizeOptions {
    ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
}

fn resize_err(e: impl std::fmt::Display) -> Error {
    Error::Other(format!("resize: {e}"))
}

pub fn resize_rgb(src: &RgbImage, w: u32, h: u32) -> Result<RgbImage> {
    if src.dimensions() == (w, h) {
        return Ok(src.clone());
    }
    let mut dst = RgbImage::new(w, h);
    Resizer::new().resize(src, &mut dst, &lanczos()).map_err(resize_err)?;
    Ok(dst)
}

pub fn resize_gray(src: &GrayImage, w: u32, h: u32) -> Result<GrayImage> {
    if src.dimensions() == (w, h) {
        return Ok(src.clone());
    }
    let mut dst = GrayImage::new(w, h);
    Resizer::new().resize(src, &mut dst, &lanczos()).map_err(resize_err)?;
    Ok(dst)
}

/// Resize keeping aspect ratio so the longest side is at most `max_side`.
pub fn fit_rgba(src: &RgbaImage, max_side: u32) -> Result<RgbaImage> {
    let (w, h) = src.dimensions();
    let s = (max_side as f32 / w.max(h) as f32).min(1.0);
    let (nw, nh) = (((w as f32 * s).round() as u32).max(1), ((h as f32 * s).round() as u32).max(1));
    if (nw, nh) == (w, h) {
        return Ok(src.clone());
    }
    let mut dst = RgbaImage::new(nw, nh);
    Resizer::new().resize(src, &mut dst, &lanczos()).map_err(resize_err)?;
    Ok(dst)
}

/// Apply `mask` as alpha to `img` (multiplied with any existing alpha).
pub fn cutout(img: &DynamicImage, mask: &GrayImage) -> RgbaImage {
    let mut out = img.to_rgba8();
    debug_assert_eq!(out.dimensions(), mask.dimensions());
    for (p, m) in out.pixels_mut().zip(mask.pixels()) {
        p[3] = ((p[3] as u16 * m[0] as u16 + 127) / 255) as u8;
    }
    out
}

/// Alpha-composite `fg` over a solid colour.
pub fn flatten(fg: &RgbaImage, bg: [u8; 3]) -> RgbImage {
    ImageBuffer::from_fn(fg.width(), fg.height(), |x, y| {
        let p = fg.get_pixel(x, y);
        let a = p[3] as u32;
        let mix = |c: u8, b: u8| ((c as u32 * a + b as u32 * (255 - a) + 127) / 255) as u8;
        Rgb([mix(p[0], bg[0]), mix(p[1], bg[1]), mix(p[2], bg[2])])
    })
}

/// Composite `fg` over a checkerboard (for previews / grids).
pub fn over_checker(fg: &RgbaImage, cell: u32) -> RgbaImage {
    let mut out = checkerboard(fg.width(), fg.height(), cell);
    for (o, p) in out.pixels_mut().zip(fg.pixels()) {
        o.blend(p);
    }
    out
}

pub fn checkerboard(w: u32, h: u32, cell: u32) -> RgbaImage {
    let cell = cell.max(1);
    ImageBuffer::from_fn(w, h, |x, y| {
        if ((x / cell) + (y / cell)) % 2 == 0 {
            Rgba([204, 204, 204, 255])
        } else {
            Rgba([255, 255, 255, 255])
        }
    })
}

/// Hard threshold a soft mask (values > t become 255).
pub fn threshold(mask: &mut GrayImage, t: u8) {
    for p in mask.pixels_mut() {
        *p = Luma([if p[0] > t { 255 } else { 0 }]);
    }
}

/// Mask as an opaque RGBA image (for display).
pub fn mask_rgba(mask: &GrayImage) -> RgbaImage {
    ImageBuffer::from_fn(mask.width(), mask.height(), |x, y| {
        let v = mask.get_pixel(x, y)[0];
        Rgba([v, v, v, 255])
    })
}

/// Parse `#rgb`, `#rrggbb`, or a few colour names.
pub fn parse_color(s: &str) -> Option<[u8; 3]> {
    match s.to_ascii_lowercase().as_str() {
        "white" => return Some([255, 255, 255]),
        "black" => return Some([0, 0, 0]),
        "gray" | "grey" => return Some([128, 128, 128]),
        "green" => return Some([0, 177, 64]),
        _ => {}
    }
    let h = s.strip_prefix('#').unwrap_or(s);
    let v = |i: usize, n: usize| u8::from_str_radix(&h[i..i + n], 16).ok();
    match h.len() {
        3 => Some([v(0, 1)? * 17, v(1, 1)? * 17, v(2, 1)? * 17]),
        6 => Some([v(0, 2)?, v(2, 2)?, v(4, 2)?]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors() {
        assert_eq!(parse_color("#fff"), Some([255, 255, 255]));
        assert_eq!(parse_color("00ff10"), Some([0, 255, 16]));
        assert_eq!(parse_color("nope"), None);
    }

    #[test]
    fn cutout_and_flatten() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 1, Rgb([200, 100, 0])));
        let mask = GrayImage::from_raw(2, 1, vec![255, 0]).unwrap();
        let c = cutout(&img, &mask);
        assert_eq!(c.get_pixel(0, 0)[3], 255);
        assert_eq!(c.get_pixel(1, 0)[3], 0);
        let f = flatten(&c, [0, 0, 255]);
        assert_eq!(f.get_pixel(0, 0).0, [200, 100, 0]);
        assert_eq!(f.get_pixel(1, 0).0, [0, 0, 255]);
    }
}
