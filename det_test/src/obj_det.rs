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

    let normalized = trimmed.replace('\'', "\"");
    let Ok(val) = serde_json::from_str::<Value>(&normalized) else {
        return Vec::new();
    };

    match val {
        Value::Object(obj) => {
            let mut map = BTreeMap::new();
            for (k, v) in obj {
                if let (Ok(idx), Some(name)) = (k.trim().parse::<usize>(), v.as_str()) {
                    map.insert(idx, name.to_string());
                }
            }
            let Some(&max_idx) = map.keys().max() else {
                return Vec::new();
            };
            (0..=max_idx)
                .map(|i| map.remove(&i).unwrap_or_else(|| i.to_string()))
                .collect()
        }
        Value::Array(arr) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(ToOwned::to_owned))
            .collect(),
        _ => Vec::new(),
    }
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
        fallback_labels: Option<Vec<String>>,
        input_size: u32,
        conf_threshold: f32,
    ) -> Result<Self> {
        let session = Session::builder(env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load YOLO ONNX model: {}", model_path.display()))?;

        let mut labels = session
            .metadata()
            .ok()
            .and_then(|m| m.custom("names").ok().flatten())
            .map(|raw| parse_metadata_names(&raw))
            .unwrap_or_default();

        if labels.is_empty() {
            print!("Class mapping not found in ONNX model, using fallback labels");
            labels = fallback_labels.unwrap_or_default();
        }

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
        let (_shape, raw) = outputs[0].try_extract_tensor::<f32>()?;

        let num_classes = self.labels.len();
        let num_preds = if num_classes + 4 > 0 {
            raw.len() / (num_classes + 4)
        } else {
            0
        };

        let sx = prepared.orig_w as f32 / self.input_size as f32;
        let sy = prepared.orig_h as f32 / self.input_size as f32;
        let mut detections = Vec::new();

        for i in 0..num_preds {
            let cx = raw[i];
            let cy = raw[num_preds + i];
            let w = raw[2 * num_preds + i];
            let h = raw[3 * num_preds + i];

            let mut best_cls = 0;
            let mut best_score = 0.0_f32;
            for c in 0..num_classes {
                let score = raw[(4 + c) * num_preds + i];
                if score > best_score {
                    best_score = score;
                    best_cls = c;
                }
            }

            if best_score >= self.conf_threshold {
                let x1 = ((cx - w * 0.5) * sx).max(0.0);
                let y1 = ((cy - h * 0.5) * sy).max(0.0);
                let x2 = ((cx + w * 0.5) * sx).min(prepared.orig_w as f32);
                let y2 = ((cy + h * 0.5) * sy).min(prepared.orig_h as f32);

                let label = self
                    .labels
                    .get(best_cls)
                    .cloned()
                    .unwrap_or_else(|| best_cls.to_string());

                detections.push(YoloDetection {
                    class: label,
                    confidence: best_score,
                    bbox: rescale_xyxy([x1, y1, x2, y2], prepared.resize_scale),
                });
            }
        }

        Ok(detections)
    }
}