use crate::preprocess::rescale_xyxy;
use crate::types::YoloDetection;
use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::Array4;
use ort::{
    environment::Environment,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use std::path::Path;
use serde_json::Value;
use std::collections::BTreeMap;

fn parse_metadata_names(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let normalized = trimmed
        .replace('{', "")
        .replace('}', "")
        .replace('\'', "\"");
    let mut map = BTreeMap::new();

    for part in normalized.split(',') {
        let mut pair = part.splitn(2, ':');
        let (Some(key), Some(value)) = (pair.next(), pair.next()) else {
            continue;
        };
        let Ok(index) = key.trim().trim_matches('"').parse::<usize>() else {
            continue;
        };
        let name = value.trim().trim_matches('"').trim();
        if !name.is_empty() {
            map.insert(index, name.to_string());
        }
    }

    if map.is_empty() {
        if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(trimmed) {
            return arr
                .into_iter()
                .filter_map(|value| value.as_str().map(ToOwned::to_owned))
                .collect();
        }
        if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(trimmed) {
            for (key, value) in obj {
                if let (Ok(index), Some(name)) = (key.parse::<usize>(), value.as_str()) {
                    map.insert(index, name.to_string());
                }
            }
        }
    }

    let Some(&max_index) = map.keys().max() else {
        return Vec::new();
    };
    (0..=max_index)
        .map(|index| map.remove(&index).unwrap_or_else(|| index.to_string()))
        .collect()
}

pub struct YoloPreparedInput {
    pub tensor: Array4<f32>,
    pub orig_w: u32,
    pub orig_h: u32,
    pub resize_scale: f32,
}

pub struct YoloDetector {
    session: Session,
    labels: Vec<String>,
    input_size: u32,
    conf_threshold: f32,
}

impl YoloDetector {
    pub fn new(
        env: &Environment,
        model_path: &Path,
        input_size: u32,
        conf_threshold: f32,
    ) -> Result<Self> {
        let session = Session::builder(env)
            .map_err(|e| anyhow::anyhow!("Failed to create YOLO session builder: {e:?}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to configure YOLO optimization: {e:?}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to configure YOLO threads: {e:?}"))?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load YOLO ONNX model: {}", model_path.display()))?;

        let mut labels = session
            .metadata()
            .ok()
            .and_then(|m| m.custom("names"))
            .map(|raw| parse_metadata_names(&raw))
            .unwrap_or_default();

        Ok(Self {
            session,
            labels,
            input_size,
            conf_threshold,
        })
    }

    pub fn prepare(&self, img: &DynamicImage, resize_scale: f32) -> YoloPreparedInput {
        let (orig_w, orig_h) = img.dimensions();
        let resized = img.resize_exact(self.input_size, self.input_size, FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let sz = self.input_size as usize;
        let mut tensor = Array4::<f32>::zeros((1, 3, sz, sz));
        for (x, y, px) in rgb.enumerate_pixels() {
            tensor[[0, 0, y as usize, x as usize]] = px[0] as f32 / 255.0;
            tensor[[0, 1, y as usize, x as usize]] = px[1] as f32 / 255.0;
            tensor[[0, 2, y as usize, x as usize]] = px[2] as f32 / 255.0;
        }

        YoloPreparedInput {
            tensor,
            orig_w,
            orig_h,
            resize_scale,
        }
    }

    pub fn detect(&mut self, prepared: YoloPreparedInput) -> Result<Vec<YoloDetection>> {
        let input_tensor = Tensor::from_array(prepared.tensor)?;
        let outputs = self.session.run(ort::inputs!["images" => input_tensor])?;
        let (shape, raw) = outputs[0].try_extract_tensor::<f32>()?;

        let num_classes = self.labels.len();
        let mut candidates = Vec::new();
        let sx = prepared.orig_w as f32 / self.input_size as f32;
        let sy = prepared.orig_h as f32 / self.input_size as f32;

        let (num_predictions, prediction_major) = match shape.as_ref() {
            [_, channels, predictions] if *channels as usize == num_classes + 4 => {
                (*predictions as usize, false)
            }
            [_, predictions, channels] if *channels as usize == num_classes + 4 => {
                (*predictions as usize, true)
            }
            _ => return Ok(Vec::new()),
        };

        for prediction in 0..num_predictions {
            let base = if prediction_major { prediction * (num_classes + 4) } else { prediction };
            let stride = if prediction_major { 1 } else { num_predictions };

            let cx = raw[base];
            let cy = raw[base + stride];
            let width = raw[base + 2 * stride];
            let height = raw[base + 3 * stride];

            let mut best_class = 0;
            let mut best_score = 0.0_f32;
            for class in 0..num_classes {
                let score = raw[base + (4 + class) * stride];
                if score > best_score {
                    best_score = score;
                    best_class = class;
                }
            }

            if best_score < self.conf_threshold {
                continue;
            }

            let x1 = ((cx - width * 0.5) * sx).max(0.0);
            let y1 = ((cy - height * 0.5) * sy).max(0.0);
            let x2 = ((cx + width * 0.5) * sx).min(prepared.orig_w as f32);
            let y2 = ((cy + height * 0.5) * sy).min(prepared.orig_h as f32);

            if x2 > x1 && y2 > y1 {
                candidates.push(YoloCandidate {
                    class_index: best_class,
                    confidence: best_score,
                    bbox: [x1, y1, x2, y2],
                });
            }
        }

        Ok(apply_nms(candidates, self.labels.as_slice(), prepared.resize_scale))
    }
}

#[derive(Clone)]
struct YoloCandidate {
    class_index: usize,
    confidence: f32,
    bbox: [f32; 4],
}

fn apply_nms(
    mut candidates: Vec<YoloCandidate>,
    labels: &[String],
    resize_scale: f32,
) -> Vec<YoloDetection> {
    const IOU_THRESHOLD: f32 = 0.45;

    candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept = Vec::new();

    while let Some(candidate) = candidates.first().cloned() {
        candidates.remove(0);
        candidates.retain(|other| {
            other.class_index != candidate.class_index
                || intersection_over_union(candidate.bbox, other.bbox) < IOU_THRESHOLD
        });

        let label = labels
            .get(candidate.class_index)
            .cloned()
            .unwrap_or_else(|| candidate.class_index.to_string());
        kept.push(YoloDetection {
            class: label,
            confidence: candidate.confidence,
            bbox: rescale_xyxy(candidate.bbox, resize_scale),
        });
    }

    kept
}

fn intersection_over_union(a: [f32; 4], b: [f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - intersection;
    if union <= 0.0 { 0.0 } else { intersection / union }
}