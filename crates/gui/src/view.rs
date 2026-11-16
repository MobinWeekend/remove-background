//! Zoomable / pannable image canvas with selectable preview background
//! and an optional A/B split slider.

use egui::{
    Color32, ColorImage, Context, Id, Pos2, Rect, Response, Sense, Stroke, TextureFilter, TextureHandle,
    TextureOptions, TextureWrapMode, Ui, Vec2, pos2, vec2,
};
use image::RgbaImage;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bg {
    Checker,
    Dark,
    Light,
    Custom(Color32),
}

impl Bg {
    pub fn solid(self) -> Option<[u8; 3]> {
        match self {
            Bg::Checker => None,
            Bg::Dark => Some([24, 24, 27]),
            Bg::Light => Some([250, 250, 250]),
            Bg::Custom(c) => Some([c.r(), c.g(), c.b()]),
        }
    }
}

/// Zoom (relative to fit) + view centre in normalized image coords.
/// Shared between compare tiles so they stay in sync.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewState {
    pub zoom: f32,
    pub center: Vec2,
}

impl Default for ViewState {
    fn default() -> Self {
        Self { zoom: 1.0, center: vec2(0.5, 0.5) }
    }
}

pub fn display_options() -> TextureOptions {
    TextureOptions {
        magnification: TextureFilter::Linear,
        minification: TextureFilter::Linear,
        wrap_mode: TextureWrapMode::ClampToEdge,
        mipmap_mode: Some(TextureFilter::Linear),
    }
}

pub fn load_texture(ctx: &Context, name: &str, img: &RgbaImage) -> TextureHandle {
    let max = ctx.input(|i| i.max_texture_side) as u32;
    let fitted;
    let img = if img.width() > max || img.height() > max {
        fitted = rembg_core::imageops::fit_rgba(img, max).expect("resize");
        &fitted
    } else {
        img
    };
    let ci = ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
    ctx.load_texture(name, ci, display_options())
}

pub fn checker_texture(ctx: &Context) -> TextureHandle {
    let (a, b) = (Color32::from_gray(204), Color32::from_gray(255));
    let ci = ColorImage::new([2, 2], vec![a, b, b, a]);
    ctx.load_texture("checker", ci, TextureOptions::NEAREST_REPEAT)
}

pub struct Layer<'a> {
    pub tex: &'a TextureHandle,
}

pub struct Split<'a> {
    /// Texture shown right of the divider.
    pub right: &'a TextureHandle,
    /// Divider position, 0..1 of the canvas width.
    pub pos: &'a mut f32,
}

/// Paint `layer` into `rect`, handling zoom (wheel / pinch), pan (drag) and
/// reset (double-click). Returns the canvas response.
#[allow(clippy::too_many_arguments)]
pub fn canvas(
    ui: &mut Ui,
    rect: Rect,
    id: Id,
    layer: Layer<'_>,
    img_size: Vec2,
    state: &mut ViewState,
    bg: Bg,
    checker: &TextureHandle,
    split: Option<Split<'_>>,
) -> Response {
    let resp = ui.interact(rect, id, Sense::click_and_drag());
    let fit = (rect.width() / img_size.x).min(rect.height() / img_size.y);

    // Divider handle gets priority over panning.
    let mut split = split;
    let mut divider_dragging = false;
    if let Some(s) = split.as_mut() {
        let x = rect.left() + *s.pos * rect.width();
        let handle = Rect::from_center_size(pos2(x, rect.center().y), vec2(14.0, rect.height()));
        let h = ui.interact(handle, id.with("divider"), Sense::drag());
        if h.hovered() || h.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }
        if h.dragged() {
            divider_dragging = true;
            if let Some(p) = h.interact_pointer_pos() {
                *s.pos = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            }
        }
    }

    if resp.hovered() {
        let (scroll, pinch, pointer) =
            ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta(), i.pointer.hover_pos()));
        let factor = pinch * (scroll * 0.0025).exp();
        if (factor - 1.0).abs() > 1e-4
            && let Some(cursor) = pointer
        {
            zoom_about(state, rect, img_size, fit, cursor, factor);
        }
    }
    if resp.dragged() && !divider_dragging {
        let disp = img_size * fit * state.zoom;
        state.center -= resp.drag_delta() / disp;
    }
    if resp.double_clicked() {
        *state = ViewState::default();
    }

    let disp = img_size * fit * state.zoom;
    let img_rect = Rect::from_min_size(rect.center() - state.center * disp, disp);
    let painter = ui.painter_at(rect);
    let visible = img_rect.intersect(rect);

    // Background behind the (transparent) cut-out only.
    match bg.solid() {
        None => {
            let cell = 10.0;
            let uv = Rect::from_min_max(
                pos2(visible.left() / (2.0 * cell), visible.top() / (2.0 * cell)),
                pos2(visible.right() / (2.0 * cell), visible.bottom() / (2.0 * cell)),
            );
            painter.image(checker.id(), visible, uv, Color32::WHITE);
        }

        Some([r, g, b]) => {
            painter.rect_filled(visible, 0.0, Color32::from_rgb(r, g, b));
        }
    }

    let full_uv = Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0));
    match split {
        None => {
            painter.image(layer.tex.id(), img_rect, full_uv, Color32::WHITE);
        }
        Some(s) => {
            let x = rect.left() + *s.pos * rect.width();
            let left = Rect::from_min_max(rect.min, pos2(x, rect.bottom()));
            let right = Rect::from_min_max(pos2(x, rect.top()), rect.max);
            ui.painter_at(left).image(layer.tex.id(), img_rect, full_uv, Color32::WHITE);
            ui.painter_at(right).image(s.right.id(), img_rect, full_uv, Color32::WHITE);
            let stroke = Stroke::new(2.0, Color32::from_rgb(80, 160, 255));
            painter.line_segment([pos2(x, rect.top()), pos2(x, rect.bottom())], stroke);
            painter.circle_filled(pos2(x, rect.center().y), 9.0, Color32::from_rgb(80, 160, 255));
            painter.text(
                pos2(x, rect.center().y),
                egui::Align2::CENTER_CENTER,
                "↔",
                egui::FontId::proportional(12.0),
                Color32::WHITE,
            );
        }
    }
    resp
}

fn zoom_about(state: &mut ViewState, rect: Rect, img: Vec2, fit: f32, cursor: Pos2, factor: f32) {
    let disp = img * fit * state.zoom;
    let img_min = rect.center() - state.center * disp;
    let p = (cursor - img_min) / disp;
    state.zoom = (state.zoom * factor).clamp(0.1, 64.0);
    let disp2 = img * fit * state.zoom;
    let img_min2 = cursor - p * disp2;
    state.center = (rect.center() - img_min2) / disp2;
}
