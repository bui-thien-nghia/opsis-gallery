use crate::preprocess::rescale_xyxy;
use crate::types::YoloDetection;
use anyhow::{Context, Result};
use image::DynamicImage;
use std::path::Path;
use ultralytics_inference::{InferenceConfig, YOLOModel};

pub struct YoloPreparedInput<'a> {
    pub img: &'a DynamicImage,
    pub resize_scale: f32,
}

pub struct YoloDetector {
    model: YOLOModel,
}

impl YoloDetector {
    pub fn new(model_path: &Path, conf_threshold: f32) -> Result<Self> {
        let config = InferenceConfig::new()
            .with_confidence(conf_threshold)
            .with_iou(0.7);
        let model = YOLOModel::load_with_config(model_path, config).with_context(|| {
            format!(
                "Failed to load Ultralytics model: {}",
                model_path.display()
            )
        })?;

        Ok(Self { model })
    }

    pub fn prepare<'a>(&self, img: &'a DynamicImage, resize_scale: f32) -> YoloPreparedInput<'a> {
        YoloPreparedInput { img, resize_scale }
    }

    pub fn detect(&mut self, prepared: YoloPreparedInput<'_>) -> Result<Vec<YoloDetection>> {
        let results = self
            .model
            .predict_image(prepared.img, String::new())
            .context("Ultralytics inference failed")?;

        let mut detections = Vec::new();
        for result in &results {
            let Some(ref boxes) = result.boxes else {
                continue;
            };
            let xyxy = boxes.xyxy();
            let confs = boxes.conf();
            let classes = boxes.cls();

            for i in 0..boxes.len() {
                let class_id = classes[i] as usize;
                let class = result
                    .names
                    .get(&class_id)
                    .cloned()
                    .unwrap_or_else(|| class_id.to_string());
                let bbox = rescale_xyxy(
                    [xyxy[[i, 0]], xyxy[[i, 1]], xyxy[[i, 2]], xyxy[[i, 3]]],
                    prepared.resize_scale,
                );
                detections.push(YoloDetection {
                    class,
                    confidence: confs[i],
                    bbox,
                });
            }
        }

        Ok(detections)
    }
}
