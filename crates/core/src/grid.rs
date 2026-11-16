//! Compose several results into one labelled comparison image.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};
use image::{Rgba, RgbaImage, imageops::overlay};

use crate::{Result, imageops};

pub struct Tile<'a> {
    pub label: String,
    pub image: &'a RgbaImage,
}

/// Background drawn behind each (transparent) tile.
#[derive(Debug, Clone, Copy)]
pub enum TileBg {
    Checker,
    Solid([u8; 3]),
}

const LABEL_H: u32 = 34;
const PAD: u32 = 8;

/// Lay tiles out in a near-square grid; each tile's longest side is `tile_side` px
/// and cells follow the first tile's aspect ratio.
pub fn compose(tiles: &[Tile<'_>], tile_side: u32, bg: TileBg) -> Result<RgbaImage> {
    let n = tiles.len().max(1) as u32;
    let (iw, ih) = tiles.first().map(|t| t.image.dimensions()).unwrap_or((1, 1));
    let aspect = iw as f32 / ih.max(1) as f32;
    let (tw, th) = if aspect >= 1.0 {
        (tile_side, ((tile_side as f32 / aspect).round() as u32).max(1))
    } else {
        (((tile_side as f32 * aspect).round() as u32).max(1), tile_side)
    };
    // Choose column count so the overall grid is roughly square.
    let cols = (1..=n)
        .min_by_key(|&c| {
            let r = n.div_ceil(c);
            let (gw, gh) = (c * (tw + PAD), r * (th + LABEL_H + PAD));
            (gw.max(gh) as f32 / gw.min(gh) as f32 * 1000.0) as u32 + (c * r - n) * 50
        })
        .unwrap_or(1);
    let rows = n.div_ceil(cols);
    let cell_w = tw + PAD;
    let cell_h = th + LABEL_H + PAD;
    let mut out = RgbaImage::from_pixel(cols * cell_w + PAD, rows * cell_h + PAD, Rgba([32, 32, 36, 255]));
    let font = FontRef::try_from_slice(epaint_default_fonts::UBUNTU_LIGHT).expect("bundled font");

    for (i, t) in tiles.iter().enumerate() {
        let (c, r) = (i as u32 % cols, i as u32 / cols);
        let (x0, y0) = (PAD + c * cell_w, PAD + r * cell_h);
        draw_text(&mut out, &font, &t.label, x0 + 2, y0 + 6, 20.0, [235, 235, 235]);
        let fitted = imageops::fit_rgba(t.image, tile_side)?;
        let backed = match bg {
            TileBg::Checker => imageops::over_checker(&fitted, 12),
            TileBg::Solid(c) => image::DynamicImage::ImageRgb8(imageops::flatten(&fitted, c)).to_rgba8(),
        };
        let ox = x0 + tw.saturating_sub(backed.width()) / 2;
        let oy = y0 + LABEL_H + th.saturating_sub(backed.height()) / 2;
        overlay(&mut out, &backed, ox as i64, oy as i64);
    }
    Ok(out)
}

fn draw_text(img: &mut RgbaImage, font: &FontRef<'_>, text: &str, x: u32, y: u32, px: f32, rgb: [u8; 3]) {
    let scaled = font.as_scaled(PxScale::from(px));
    let mut caret = x as f32;
    let baseline = y as f32 + scaled.ascent();
    let mut prev = None;
    for ch in text.chars() {
        let id = scaled.glyph_id(ch);
        if let Some(p) = prev {
            caret += scaled.kern(p, id);
        }
        let g = id.with_scale_and_position(scaled.scale(), point(caret, baseline));
        caret += scaled.h_advance(id);
        prev = Some(id);
        if let Some(og) = font.outline_glyph(g) {
            let b = og.px_bounds();
            og.draw(|gx, gy, cov| {
                let (px_, py_) = (b.min.x as i32 + gx as i32, b.min.y as i32 + gy as i32);
                if px_ >= 0 && py_ >= 0 && (px_ as u32) < img.width() && (py_ as u32) < img.height() {
                    let p = img.get_pixel_mut(px_ as u32, py_ as u32);
                    for k in 0..3 {
                        p[k] = (p[k] as f32 * (1.0 - cov) + rgb[k] as f32 * cov) as u8;
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_dims() {
        let a = RgbaImage::from_pixel(100, 50, Rgba([255, 0, 0, 255]));
        let tiles: Vec<_> = (0..5).map(|i| Tile { label: format!("m{i} · 12 ms"), image: &a }).collect();
        let g = compose(&tiles, 64, TileBg::Checker).unwrap();
        // 2:1 tiles -> 64x32 cells; grid stays roughly square
        let (w, h) = g.dimensions();
        assert!(w.max(h) as f32 / w.min(h) as f32 <= 2.0, "{w}x{h}");
        assert_eq!((w - PAD) % (64 + PAD), 0);
        assert_eq!((h - PAD) % (32 + LABEL_H + PAD), 0);
    }
}
