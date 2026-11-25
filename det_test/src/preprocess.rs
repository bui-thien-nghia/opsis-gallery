use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use std::path::Path;

pub fn load_image(path: &Path) -> Result<DynamicImage> {
    if !path.exists() {
        bail!("Image file does not exist: {}", path.display());
    }

    let reader = image::ImageReader::open(path)
        .with_context(|| format!("Failed to open image: {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("Failed to detect image format: {}", path.display()))?;

    reader
        .decode()
        .with_context(|| format!("Failed to decode image: {}", path.display()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_resize_leaves_small_image_unchanged() {
        let img = DynamicImage::new_rgb8(320, 240);
        let (resized, scale) = adaptive_resize(img, 1600);
        assert_eq!(resized.dimensions(), (320, 240));
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn adaptive_resize_scales_down_large_image() {
        let img = DynamicImage::new_rgb8(3200, 1600);
        let (resized, scale) = adaptive_resize(img, 1600);
        assert_eq!(resized.dimensions(), (1600, 800));
        assert!((scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rescale_helpers_invert_resize_scale() {
        assert_eq!(
            rescale_xyxy([10.0, 20.0, 30.0, 40.0], 0.5),
            [20.0, 40.0, 60.0, 80.0]
        );
        assert_eq!(
            rescale_quad([[5.0, 10.0], [15.0, 10.0], [15.0, 20.0], [5.0, 20.0]], 0.5),
            [[10, 20], [30, 20], [30, 40], [10, 40]]
        );
    }

    #[test]
    fn load_image_errors_on_missing_file() {
        assert!(load_image(Path::new("nonexistent_test_image_12345.png")).is_err());
    }
}
