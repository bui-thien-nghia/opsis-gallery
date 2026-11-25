use crate::preprocess::rescale_xyxy;
use crate::types::YoloDetection;
use anyhow::{Context, Result};
use image::{DynamicImage, imageops::FilterType};
use ndarray::{Array3, Axis};
use ort::{
    environment::Environment,
    session::{
        Session,
        builder::{AutoDevicePolicy, GraphOptimizationLevel},
    },
    value::ValueType,
};
use std::{collections::BTreeMap, path::Path};

pub struct YoloPreparedInput<'a> {
    pub img: &'a DynamicImage,
    pub resize_scale: f32,
}

pub struct YoloDetector {
    session: Session,
    labels: Vec<String>,
    input_size: u32,
    conf_threshold: f32,
    iou_threshold: f32,
}

fn parse_names(raw: &str) -> Result<Vec<String>> {
    let inner = raw.trim().trim_start_matches('{').trim_end_matches('}');
    let mut entries = BTreeMap::new();

    for entry in inner.split(',') {
        let (key, value) = entry
            .split_once(':')
            .context("Invalid YOLO metadata names entry")?;
        let index = key
            .trim()
            .parse::<usize>()
            .context("Invalid YOLO class index")?;
        let mut name = value.trim();
        if name.starts_with(['\'', '"']) && name.ends_with(['\'', '"']) {
            name = &name[1..name.len().saturating_sub(1)];
        }
        entries.insert(index, name.trim().replace("\\'", "'").replace("\\\"", "\""));
    }

    anyhow::ensure!(!entries.is_empty(), "YOLO metadata contains no class names");
    for (expected, index) in (0..entries.len()).zip(entries.keys()) {
        anyhow::ensure!(
            expected == *index,
            "YOLO metadata class indices are not contiguous"
        );
    }

    Ok(entries.into_values().collect())
}

fn input_size_from_session(session: &Session) -> Result<u32> {
    let input = session
        .inputs()
        .first()
        .context("YOLO model has no input tensor")?;
    let ValueType::Tensor { shape, .. } = input.dtype() else {
        anyhow::bail!("YOLO model input is not a tensor")
    };

    anyhow::ensure!(
        shape.len() == 4,
        "Unexpected YOLO input rank: {}",
        shape.len()
    );
    let height = shape[2];
    let width = shape[3];
    anyhow::ensure!(
        height > 0 && width > 0,
        "YOLO input dimensions must be fixed"
    );
    anyhow::ensure!(
        height == width,
        "YOLO requires a square input, got {width}x{height}"
    );

    u32::try_from(width).context("Invalid YOLO input width")
}

impl YoloDetector {
    pub fn new(env: &Environment, model_path: &Path, conf_threshold: f32) -> Result<Self> {
        let session = Session::builder(env)
            .map_err(|e| anyhow::anyhow!("Failed to create YOLO session builder: {e:?}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to configure YOLO optimization: {e:?}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to configure YOLO threads: {e:?}"))?
            .with_auto_device(AutoDevicePolicy::PreferCPU)
            .map_err(|e| anyhow::anyhow!("Failed to set YOLO device policy: {e:?}"))?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load YOLO ONNX model: {}", model_path.display()))?;

        let input_size = input_size_from_session(&session)?;
        let labels = {
            let metadata = session
                .metadata()
                .context("Failed to read YOLO ONNX metadata")?;
            let names = metadata
                .custom("names")
                .context("YOLO ONNX metadata does not contain the 'names' class map")?;
            parse_names(&names)?
        };

        session
            .outputs()
            .iter()
            .find(|output| output.name() == "output0")
            .context("YOLO ONNX model has no 'output0' tensor")?;

        let output_shape = session
            .outputs()
            .iter()
            .find(|output| output.name() == "output0")
            .and_then(|output| match output.dtype() {
                ValueType::Tensor { shape, .. } => Some(shape.clone()),
                _ => None,
            })
            .context("YOLO 'output0' is not a tensor")?;

        let class_count = if output_shape.len() == 3 {
            let candidate_a = output_shape[1].saturating_sub(4) as usize;
            let candidate_b = output_shape[2].saturating_sub(4) as usize;
            if candidate_a == labels.len() {
                candidate_a
            } else if candidate_b == labels.len() {
                candidate_b
            } else {
                anyhow::bail!(
                    "YOLO metadata names count ({}) does not match output shape {:?}",
                    labels.len(),
                    output_shape,
                )
            }
        } else {
            anyhow::bail!("Unexpected YOLO output rank: {}", output_shape.len())
        };
        anyhow::ensure!(
            class_count == labels.len(),
            "YOLO metadata names count ({}) does not match output class count ({})",
            labels.len(),
            class_count,
        );

        Ok(Self {
            session,
            labels,
            input_size,
            conf_threshold,
            iou_threshold: 0.7,
        })
    }

    pub fn prepare<'a>(&self, img: &'a DynamicImage, resize_scale: f32) -> YoloPreparedInput<'a> {
        YoloPreparedInput { img, resize_scale }
    }

    pub fn detect(&mut self, prepared: YoloPreparedInput<'_>) -> Result<Vec<YoloDetection>> {
        let size = self.input_size as u32;
        let image = prepared
            .img
            .resize_exact(size, size, FilterType::CatmullRom)
            .to_rgb8();
        let mut tensor = Array3::<f32>::zeros((3, size as usize, size as usize));
        for y in 0..size as usize {
            for x in 0..size as usize {
                let pixel = image.get_pixel(x as u32, y as u32);
                tensor[[0, y, x]] = pixel[0] as f32 / 255.0;
                tensor[[1, y, x]] = pixel[1] as f32 / 255.0;
                tensor[[2, y, x]] = pixel[2] as f32 / 255.0;
            }
        }

        let input = tensor.insert_axis(Axis(0));
        let outputs = self
            .session
            .run(
                ort::inputs!["images" => ort::value::TensorRef::from_array_view(input.view())
                .context("Failed to create YOLO input tensor")?],
            )
            .context("YOLO ONNX inference failed")?;
        let output = outputs
            .get("output0")
            .context("YOLO ONNX model returned no 'output0' tensor")?
            .try_extract_array::<f32>()
            .context("Failed to extract YOLO output tensor")?
            .into_owned();

        let class_count = self.labels.len();
        let (detections, raw_width, raw_height) = (
            decode_yolo_output(&output, class_count, self.conf_threshold),
            prepared.img.width() as f32,
            prepared.img.height() as f32,
        );

        let mut boxes = Vec::with_capacity(detections.len());
        for (class_id, confidence, xc, yc, w, h) in detections {
            let width_scale = raw_width / self.input_size as f32;
            let height_scale = raw_height / self.input_size as f32;
            let bbox = rescale_xyxy(
                [
                    (xc - w / 2.0) * width_scale,
                    (yc - h / 2.0) * height_scale,
                    (xc + w / 2.0) * width_scale,
                    (yc + h / 2.0) * height_scale,
                ],
                prepared.resize_scale,
            );
            boxes.push(YoloDetection {
                class: self.labels[class_id].clone(),
                confidence,
                bbox,
            });
        }

        Ok(non_maximum_suppression(boxes, self.iou_threshold))
    }
}

fn decode_yolo_output(
    output: &ndarray::ArrayD<f32>,
    class_count: usize,
    threshold: f32,
) -> Vec<(usize, f32, f32, f32, f32, f32)> {
    let shape = output.shape();
    if shape.len() != 3 || shape[0] != 1 {
        return Vec::new();
    }

    let mut detections = Vec::new();
    let expected_columns = 4 + class_count;

    // Ultralytics exports use [1, 4+classes, anchors].
    if shape[1] == expected_columns {
        let predictions = output.index_axis(Axis(0), 0);
        for anchor in 0..shape[2] {
            let values: Vec<f32> = predictions
                .axis_iter(Axis(0))
                .map(|channel| channel[anchor])
                .collect();
            if let Some((class_id, confidence)) = best_class(&values[4..], threshold) {
                detections.push((
                    class_id, confidence, values[0], values[1], values[2], values[3],
                ));
            }
        }
    // Some compatible exports use [1, anchors, 4+classes].
    } else if shape[2] == expected_columns {
        let predictions = output.index_axis(Axis(0), 0);
        for prediction in predictions.axis_iter(Axis(0)) {
            let values: Vec<f32> = prediction.iter().copied().collect();
            if let Some((class_id, confidence)) = best_class(&values[4..], threshold) {
                detections.push((
                    class_id, confidence, values[0], values[1], values[2], values[3],
                ));
            }
        }
    }
    detections
}

fn best_class(probabilities: &[f32], threshold: f32) -> Option<(usize, f32)> {
    probabilities
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .filter(|(_, confidence)| *confidence >= threshold)
}

fn non_maximum_suppression(
    mut detections: Vec<YoloDetection>,
    iou_threshold: f32,
) -> Vec<YoloDetection> {
    detections.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept = Vec::new();

    for detection in detections {
        let overlaps = kept.iter().any(|selected: &YoloDetection| {
            if selected.class != detection.class {
                return false;
            }
            let [ax1, ay1, ax2, ay2] = selected.bbox;
            let [bx1, by1, bx2, by2] = detection.bbox;
            let x1 = ax1.max(bx1);
            let y1 = ay1.max(by1);
            let x2 = ax2.min(bx2);
            let y2 = ay2.min(by2);
            let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
            let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
            let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
            let union = area_a + area_b - intersection;
            union > 0.0 && intersection / union >= iou_threshold
        });
        if !overlaps {
            kept.push(detection);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ultralytics_names_metadata() {
        let labels = parse_names("{0: 'Person', 1: 'Sneakers', 2: 'Chair'}").unwrap();
        assert_eq!(labels, vec!["Person", "Sneakers", "Chair"]);
    }

    #[test]
    fn rejects_gapped_names_metadata() {
        assert!(parse_names("{0: 'Person', 2: 'Chair'}").is_err());
    }

    #[test]
    fn decodes_channel_major_output_layout() {
        let mut output = ndarray::Array3::<f32>::zeros((1, 6, 1));
        output[[0, 0, 0]] = 50.0;
        output[[0, 1, 0]] = 60.0;
        output[[0, 2, 0]] = 20.0;
        output[[0, 3, 0]] = 30.0;
        output[[0, 4, 0]] = 0.9;
        output[[0, 5, 0]] = 0.1;

        let detections = decode_yolo_output(&output.into_dyn(), 2, 0.5);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].0, 0);
        assert_eq!(detections[0].1, 0.9);
    }

    #[test]
    fn decodes_anchor_major_output_layout() {
        let output = ndarray::arr3(&[[
            [50.0, 60.0, 20.0, 30.0, 0.1, 0.8],
            [10.0, 10.0, 5.0, 5.0, 0.9, 0.1],
        ]]);

        let detections = decode_yolo_output(&output.into_dyn(), 2, 0.5);
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].0, 1);
        assert_eq!(detections[1].0, 0);
    }
}
