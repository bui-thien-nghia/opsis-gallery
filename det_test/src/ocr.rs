use crate::preprocess::rescale_quad;
use crate::types::OcrDetection;
use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::Array4;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use std::path::Path;

const DET_SIZE: usize = 640;
const REC_WIDTH: usize = 100;
const REC_HEIGHT: usize = 32;

pub struct OcrPreparedDetection {
    pub det_tensor: Array4<f32>,
    pub orig_w: u32,
    pub orig_h: u32,
    pub resize_scale: f32,
}

pub struct OcrPreparedRecognition {
    pub rec_tensor: Array4<f32>,
    pub box_coords: [u32; 4],
    pub resize_scale: f32,
}

pub struct OcrPipeline {
    detector: Session,
    recognizer: Session,
    charset: Vec<char>,
    conf_threshold: f32,
}

impl OcrPipeline {
    pub fn new(
        detector_path: &Path,
        recognizer_path: &Path,
        charset: Vec<char>,
        conf_threshold: f32,
    ) -> Result<Self> {
        let detector = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(detector_path)
            .with_context(|| {
                format!(
                    "Failed to load CRAFT ONNX detector: {}",
                    detector_path.display()
                )
            })?;

        let recognizer = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(recognizer_path)
            .with_context(|| {
                format!(
                    "Failed to load CRNN ONNX recognizer: {}",
                    recognizer_path.display()
                )
            })?;

        Ok(Self {
            detector,
            recognizer,
            charset,
            conf_threshold,
        })
    }

    pub fn prepare_detection(&self, img: &DynamicImage, resize_scale: f32) -> OcrPreparedDetection {
        let (orig_w, orig_h) = img.dimensions();
        let det_resized = img.resize_exact(DET_SIZE as u32, DET_SIZE as u32, FilterType::Triangle);
        let det_rgb = det_resized.to_rgb8();

        let mut det_tensor = Array4::<f32>::zeros((1, 3, DET_SIZE, DET_SIZE));
        for (x, y, px) in det_rgb.enumerate_pixels() {
            det_tensor[[0, 0, y as usize, x as usize]] = px[0] as f32 / 255.0;
            det_tensor[[0, 1, y as usize, x as usize]] = px[1] as f32 / 255.0;
            det_tensor[[0, 2, y as usize, x as usize]] = px[2] as f32 / 255.0;
        }

        OcrPreparedDetection {
            det_tensor,
            orig_w,
            orig_h,
            resize_scale,
        }
    }

    pub fn detect_boxes(&mut self, prepared: OcrPreparedDetection) -> Result<(Vec<[u32; 4]>, f32)> {
        if prepared.orig_w == 0 || prepared.orig_h == 0 {
            return Ok((Vec::new(), prepared.resize_scale));
        }

        let det_out = self
            .detector
            .run(ort::inputs!["input" => Tensor::from_array(prepared.det_tensor)?])?;
        let (det_shape, det_data) = det_out[0].try_extract_tensor::<f32>()?;

        let (map_h, map_w) = match det_shape.as_ref() {
            [_, _, h, w] => (*h as usize, *w as usize),
            [_, h, w] => (*h as usize, *w as usize),
            _ => (DET_SIZE / 2, DET_SIZE / 2),
        };

        let boxes = extract_boxes(
            &det_data[..map_h * map_w],
            map_w,
            map_h,
            prepared.orig_w,
            prepared.orig_h,
        );
        Ok((boxes, prepared.resize_scale))
    }

    pub fn prepare_recognitions(
        &self,
        img: &DynamicImage,
        boxes: &[[u32; 4]],
        resize_scale: f32,
    ) -> Vec<OcrPreparedRecognition> {
        let mut items = Vec::new();

        for &[x1, y1, x2, y2] in boxes {
            let bw = x2.saturating_sub(x1);
            let bh = y2.saturating_sub(y1);
            if bw == 0 || bh == 0 {
                continue;
            }

            let crop = img.crop_imm(x1, y1, bw, bh);
            let gray = crop
                .resize_exact(REC_WIDTH as u32, REC_HEIGHT as u32, FilterType::Lanczos3)
                .to_luma8();

            let mut rec_tensor = Array4::<f32>::zeros((1, 1, REC_HEIGHT, REC_WIDTH));
            for (x, y, px) in gray.enumerate_pixels() {
                rec_tensor[[0, 0, y as usize, x as usize]] = px[0] as f32 / 255.0;
            }

            items.push(OcrPreparedRecognition {
                rec_tensor,
                box_coords: [x1, y1, x2, y2],
                resize_scale,
            });
        }

        items
    }

    pub fn recognize_boxes(
        &mut self,
        prepared_items: Vec<OcrPreparedRecognition>,
    ) -> Result<Vec<OcrDetection>> {
        let mut results = Vec::new();

        for item in prepared_items {
            let rec_out = self
                .recognizer
                .run(ort::inputs!["input" => Tensor::from_array(item.rec_tensor)?])?;
            let (rec_shape, rec_data) = rec_out[0].try_extract_tensor::<f32>()?;

            let (steps, classes) = match rec_shape.as_ref() {
                [_, s, c] => (*s as usize, *c as usize),
                [s, c] => (*s as usize, *c as usize),
                _ => continue,
            };

            let (text, conf) = decode_ctc(rec_data, steps, classes, &self.charset);
            if !text.is_empty() && conf >= self.conf_threshold {
                let [x1, y1, x2, y2] = item.box_coords;
                let quad = [
                    [x1 as f32, y1 as f32],
                    [x2 as f32, y1 as f32],
                    [x2 as f32, y2 as f32],
                    [x1 as f32, y2 as f32],
                ];
                results.push(OcrDetection {
                    text,
                    confidence: conf,
                    bbox: rescale_quad(quad, item.resize_scale),
                });
            }
        }

        Ok(results)
    }
}

fn extract_boxes(
    region_map: &[f32],
    map_w: usize,
    map_h: usize,
    orig_w: u32,
    orig_h: u32,
) -> Vec<[u32; 4]> {
    let thresh = 0.5_f32;
    let mut visited = vec![false; map_w * map_h];
    let mut boxes = Vec::new();
    let sx = orig_w as f32 / map_w as f32;
    let sy = orig_h as f32 / map_h as f32;

    for y in 0..map_h {
        for x in 0..map_w {
            let idx = y * map_w + x;
            if visited[idx] || region_map.get(idx).copied().unwrap_or(0.0) < thresh {
                continue;
            }

            let mut stack = vec![(x, y)];
            visited[idx] = true;
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (x, y, x, y);

            while let Some((cx, cy)) = stack.pop() {
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);

                for (nx, ny) in [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ] {
                    if nx < map_w && ny < map_h {
                        let nidx = ny * map_w + nx;
                        if !visited[nidx] && region_map.get(nidx).copied().unwrap_or(0.0) >= thresh {
                            visited[nidx] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }

            let x1 = ((min_x as f32 * sx).floor() as u32).min(orig_w.saturating_sub(1));
            let y1 = ((min_y as f32 * sy).floor() as u32).min(orig_h.saturating_sub(1));
            let x2 = (((max_x + 1) as f32 * sx).ceil() as u32).min(orig_w);
            let y2 = (((max_y + 1) as f32 * sy).ceil() as u32).min(orig_h);

            if x2 > x1 && y2 > y1 {
                boxes.push([x1, y1, x2, y2]);
            }
        }
    }

    boxes
}

fn decode_ctc(data: &[f32], steps: usize, classes: usize, charset: &[char]) -> (String, f32) {
    let mut chars = String::new();
    let mut conf_sum = 0.0_f32;
    let mut conf_count = 0_usize;
    let mut prev_idx = 0_usize;

    for t in 0..steps {
        let base = t * classes;
        let mut best_idx = 0_usize;
        let mut best_val = f32::NEG_INFINITY;

        for c in 0..classes {
            let v = data.get(base + c).copied().unwrap_or(f32::NEG_INFINITY);
            if v > best_val {
                best_val = v;
                best_idx = c;
            }
        }

        if best_idx != 0 && best_idx != prev_idx {
            let char_idx = best_idx.saturating_sub(1);
            if let Some(&ch) = charset.get(char_idx) {
                chars.push(ch);
                conf_sum += best_val.clamp(0.0, 1.0);
                conf_count += 1;
            }
        }
        prev_idx = best_idx;
    }

    let avg_conf = if conf_count > 0 {
        conf_sum / conf_count as f32
    } else {
        0.0
    };
    (chars, avg_conf)
}