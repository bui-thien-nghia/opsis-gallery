use anyhow::{Context, Result, bail};
use exif::{In, Reader, Tag, Value};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

pub fn collect_image_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir)
        .with_context(|| format!("Failed to read image directory: {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase());
            if matches!(
                ext.as_deref(),
                Some(
                    "jpg"
                        | "jpeg"
                        | "png"
                        | "webp"
                        | "bmp"
                        | "tif"
                        | "tiff"
                        | "gif"
                        | "HEIC"
                        | "xmp"
                )
            ) {
                paths.push(path);
            }
        }
    }
    paths.sort();
    Ok(paths)
}

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

pub fn formatted_image_date(path: &Path) -> String {
    let Ok(file) = File::open(path) else {
        return String::new();
    };
    let Ok(exif) = Reader::new().read_from_container(&mut BufReader::new(file)) else {
        return String::new();
    };

    let date = [Tag::DateTimeOriginal, Tag::DateTimeDigitized, Tag::DateTime]
        .into_iter()
        .find_map(|tag| {
            let field = exif.get_field(tag, In::PRIMARY)?;
            let Value::Ascii(values) = &field.value else {
                return None;
            };
            let bytes = values.first()?;
            let raw = std::str::from_utf8(bytes)
                .ok()?
                .trim_end_matches('\0')
                .trim();
            parse_image_date(raw)
        });

    date.map_or_else(String::new, |parts| format_date_parts(parts))
}

fn parse_image_date(raw: &str) -> Option<(u32, u32, u32)> {
    let date = raw.split([' ', 'T']).next()?;
    let separator = if date.contains(':') { ':' } else { '-' };
    let mut parts = date.split(separator);
    let year = parts.next()?.parse::<u32>().ok()?;
    let month = parts.next()?.parse::<u32>().ok()?;
    let day = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) {
        return None;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if day == 0 || day > days_in_month {
        return None;
    }
    Some((year, month, day))
}

fn format_date_parts((year, month, day): (u32, u32, u32)) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let suffix = match day % 100 {
        11..=13 => "th",
        _ => match day % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        },
    };
    format!("{} {}{}, {}", MONTHS[month as usize - 1], day, suffix, year)
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

    #[test]
    fn image_date_formats_ordinals_and_iso_dates() {
        assert_eq!(
            parse_image_date("2026:10:20 12:30:00"),
            Some((2026, 10, 20))
        );
        assert_eq!(format_date_parts((2026, 10, 20)), "October 20th, 2026");
        assert_eq!(
            parse_image_date("2026-10-21T12:30:00"),
            Some((2026, 10, 21))
        );
        assert_eq!(parse_image_date("2025:02:29 12:30:00"), None);
        assert_eq!(parse_image_date("2024:02:29 12:30:00"), Some((2024, 2, 29)));
        assert_eq!(format_date_parts((2026, 1, 1)), "January 1st, 2026");
        assert_eq!(format_date_parts((2026, 2, 2)), "February 2nd, 2026");
        assert_eq!(format_date_parts((2026, 3, 3)), "March 3rd, 2026");
        assert_eq!(format_date_parts((2026, 4, 11)), "April 11th, 2026");
    }
}
