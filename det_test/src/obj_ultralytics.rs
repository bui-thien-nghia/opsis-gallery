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
        let model_str = model_path
            .to_str()
            .context("YOLO model path is not valid UTF-8")?;

        let config = InferenceConfig::new()
            .with_confidence(conf_threshold)
            .with_iou(0.45);

        let model = YOLOModel::load_with_config(model_str, config)
            .with_context(|| format!("Failed to load YOLO model: {}", model_path.display()))?;

        Ok(Self { model })
    }

    pub fn prepare<'a>(&self, img: &'a DynamicImage, resize_scale: f32) -> YoloPreparedInput<'a> {
        YoloPreparedInput { img, resize_scale }
    }

    pub fn detect(&mut self, prepared: YoloPreparedInput<'_>) -> Result<Vec<YoloDetection>> {
        let results = self
            .model
            .predict_image(prepared.img, String::new())
            .context("ultralytics-inference prediction failed")?;

        let mut detections = Vec::new();
        let Some(result) = results.into_iter().next() else {
            return Ok(detections);
        };

        if let Some(boxes) = result.boxes {
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

                let raw_box = [xyxy[[i, 0]], xyxy[[i, 1]], xyxy[[i, 2]], xyxy[[i, 3]]];
                detections.push(YoloDetection {
                    class,
                    confidence: confs[i],
                    bbox: rescale_xyxy(raw_box, prepared.resize_scale),
                });
            }
        }

        Ok(detections)
    }
}
