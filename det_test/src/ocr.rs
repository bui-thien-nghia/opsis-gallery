use crate::preprocess::rescale_quad;
use crate::types::OcrDetection;
use anyhow::{Context, Result};
use image::DynamicImage;
use oar_ocr::prelude::*;
use std::path::Path;

pub struct OcrPipeline {
    engine: OAROCR,
    conf_threshold: f32,
}

impl OcrPipeline {
    pub fn new(
        detector_path: &Path,
        recognizer_path: &Path,
        dict_path: &Path,
        conf_threshold: f32,
    ) -> Result<Self> {
        let engine = OAROCRBuilder::new(detector_path, recognizer_path, dict_path)
            .build()
            .with_context(|| {
                format!(
                    "Failed to initialize OAR-OCR pipeline with det={}, rec={}, dict={}",
                    detector_path.display(),
                    recognizer_path.display(),
                    dict_path.display()
                )
            })?;

        Ok(Self {
            engine,
            conf_threshold,
        })
    }

    pub fn recognize(&self, img: &DynamicImage, resize_scale: f32) -> Result<Vec<OcrDetection>> {
        let rgb = img.to_rgb8();
        let results = self
            .engine
            .predict(vec![rgb])
            .context("OAR-OCR prediction failed")?;

        let mut detections = Vec::new();
        let Some(page_result) = results.into_iter().next() else {
            return Ok(detections);
        };

        for region in &page_result.text_regions {
            let Some((text, confidence)) = region.text_with_confidence() else {
                continue;
            };
            let trimmed = text.trim();
            if trimmed.is_empty() || confidence < self.conf_threshold {
                continue;
            }

            let quad = extract_quad(region);
            detections.push(OcrDetection {
                text: trimmed.to_string(),
                confidence,
                bbox: rescale_quad(quad, resize_scale),
            });
        }

        Ok(detections)
    }
}

fn extract_quad(region: &TextRegion) -> [[f32; 2]; 4] {
    let pts = &region.bounding_box.points;
    if pts.len() >= 4 {
        [
            [pts[0].x as f32, pts[0].y as f32],
            [pts[1].x as f32, pts[1].y as f32],
            [pts[2].x as f32, pts[2].y as f32],
            [pts[3].x as f32, pts[3].y as f32],
        ]
    } else if let Some(first) = pts.first() {
        let x = first.x as f32;
        let y = first.y as f32;
        [[x, y], [x, y], [x, y], [x, y]]
    } else {
        [[0.0, 0.0]; 4]
    }
}
