use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use std::path::Path;

pub fn load_image(path: &Path) -> Result<DynamicImage> {
    if !path.exists() {
        bail!("Image file does not exist: {}", path.display());
    }
    image::open(path).with_context(|| format!("Failed to decode image: {}", path.display()))
}

pub fn adaptive_resize(img: DynamicImage, max_dim: u32) -> (DynamicImage, f32) {
    let (w, h) = img.dimensions();
    let max_side = w.max(h);

    if max_side <= max_dim || max_side == 0 {
        return (img, 1.0);
    }

    let scale = max_dim as f32 / max_side as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);

    let resized = img.resize_exact(new_w, new_h, FilterType::Lanczos3);
    (resized, scale)
}

pub fn rescale_xyxy(bbox: [f32; 4], scale: f32) -> [f32; 4] {
    let inv = 1.0 / scale;
    [bbox[0] * inv, bbox[1] * inv, bbox[2] * inv, bbox[3] * inv]
}

pub fn rescale_quad(bbox: [[f32; 2]; 4], scale: f32) -> [[i32; 2]; 4] {
    let inv = 1.0 / scale;
    bbox.map(|[x, y]| [(x * inv).round() as i32, (y * inv).round() as i32])
}
