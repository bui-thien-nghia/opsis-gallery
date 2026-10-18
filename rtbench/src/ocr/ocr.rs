use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;
use std::fs;

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x_min: u32,
    pub y_min: u32,
    pub x_max: u32,
    pub y_max: u32,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct OcrResult {
    pub bbox: BoundingBox,
    pub text: String,
    pub confidence: f32,
}

pub struct DetectionEngine {
    session: Session,
    target_size: (u32, u32),
    det_thresh: f32,
    box_thresh: f32,
}

impl DetectionEngine {
    pub fn new<P: AsRef<Path>>(model_path: P, target_w: u32, target_h: u32) -> Result<Self> {
        let session = Session::builder()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .with_intra_threads(4)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .commit_from_file(model_path)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;

        Ok(Self {
            session,
            target_size: (target_w, target_h),
            det_thresh: 0.3,
            box_thresh: 0.5,
        })
    }

    fn preprocess(&self, img: &DynamicImage) -> (Array4<f32>, f32, f32) {
        let (orig_w, orig_h) = img.dimensions();
        let (target_w, target_h) = self.target_size;
        let scale_w = orig_w as f32 / target_w as f32;
        let scale_h = orig_h as f32 / target_h as f32;
        let resized = img.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let mean = [0.485f32, 0.456f32, 0.406f32];
        let std = [0.229f32, 0.224f32, 0.225f32];
        let mut tensor = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));

        for y in 0..target_h as usize {
            for x in 0..target_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let value = pixel[c] as f32 / 255.0;
                    tensor[[0, c, y, x]] = (value - mean[c]) / std[c];
                }
            }
        }

        (tensor, scale_w, scale_h)
    }

    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<BoundingBox>> {
        let (tensor, scale_w, scale_h) = self.preprocess(img);
        let inputs = ort::inputs![TensorRef::from_array_view(&tensor)?];
        let outputs = self.session.run(inputs)?;
        let (_, prob_map) = outputs[0].try_extract_tensor::<f32>()?;
        let target_h = self.target_size.1 as usize;
        let target_w = self.target_size.0 as usize;
        let mut visited = vec![vec![false; target_w]; target_h];
        let mut boxes = Vec::new();

        for y in 0..target_h {
            for x in 0..target_w {
                let prob = prob_map[y * target_w + x];
                if prob > self.det_thresh && !visited[y][x] {
                    let mut min_x = x;
                    let mut max_x = x;
                    let mut min_y = y;
                    let mut max_y = y;
                    let mut sum_prob = 0.0f32;
                    let mut count = 0;
                    let mut queue = vec![(x, y)];
                    visited[y][x] = true;

                    while let Some((cx, cy)) = queue.pop() {
                        sum_prob += prob_map[cy * target_w + cx];
                        count += 1;
                        min_x = min_x.min(cx);
                        max_x = max_x.max(cx);
                        min_y = min_y.min(cy);
                        max_y = max_y.max(cy);
                        let neighbors = [
                            (cx.wrapping_sub(1), cy),
                            (cx + 1, cy),
                            (cx, cy.wrapping_sub(1)),
                            (cx, cy + 1),
                        ];

                        for (nx, ny) in neighbors {
                            if nx < target_w
                                && ny < target_h
                                && !visited[ny][nx]
                                && prob_map[ny * target_w + nx] > self.det_thresh
                            {
                                visited[ny][nx] = true;
                                queue.push((nx, ny));
                            }
                        }
                    }

                    let avg_score = sum_prob / count as f32;
                    if avg_score >= self.box_thresh
                        && max_x - min_x > 4
                        && max_y - min_y > 4
                    {
                        boxes.push(BoundingBox {
                            x_min: (min_x as f32 * scale_w) as u32,
                            y_min: (min_y as f32 * scale_h) as u32,
                            x_max: (max_x as f32 * scale_w) as u32,
                            y_max: (max_y as f32 * scale_h) as u32,
                            score: avg_score,
                        });
                    }
                }
            }
        }

        Ok(boxes)
    }
}

pub struct RecognitionEngine {
    session: Session,
    vocab: Vec<String>,
    target_height: u32,
    max_width: u32,
}

impl RecognitionEngine {
    pub fn new<P: AsRef<Path>>(model_path: P, keys_path: P) -> Result<Self> {
        let session = Session::builder()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .with_intra_threads(4)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .commit_from_file(model_path)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let file = File::open(keys_path).context("Failed to open dictionary file")?;
        let reader = BufReader::new(file);
        let mut vocab = vec!["<blank>".to_string()];
        for line in reader.lines() {
            vocab.push(line?);
        }
        vocab.push(" ".to_string());

        Ok(Self {
            session,
            vocab,
            target_height: 48,
            max_width: 320,
        })
    }

    fn preprocess(&self, crop: &DynamicImage) -> Array4<f32> {
        let (w, h) = crop.dimensions();
        let aspect_ratio = w as f32 / h as f32;
        let scaled_w = ((self.target_height as f32 * aspect_ratio) as u32).min(self.max_width);
        let resized = crop.resize_exact(scaled_w, self.target_height, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let mut tensor = Array4::<f32>::zeros((1, 3, self.target_height as usize, self.max_width as usize));

        for y in 0..self.target_height as usize {
            for x in 0..scaled_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }

        tensor
    }

    fn ctc_decode(&self, logits: &[f32], time_steps: usize, vocab_size: usize) -> (String, f32) {
        let mut text = String::new();
        let mut last_idx = 0usize;
        let mut conf_sum = 0.0f32;
        let mut count = 0;

        for t in 0..time_steps {
            let step_logits = &logits[t * vocab_size..(t + 1) * vocab_size];
            let (max_idx, &max_val) = step_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let exp_sum: f32 = step_logits.iter().map(|value| (value - max_val).exp()).sum();
            let prob = 1.0 / exp_sum;

            if max_idx != 0 && max_idx != last_idx {
                if let Some(ch) = self.vocab.get(max_idx) {
                    text.push_str(ch);
                    conf_sum += prob;
                    count += 1;
                }
            }
            last_idx = max_idx;
        }

        let confidence = if count > 0 { conf_sum / count as f32 } else { 0.0 };
        (text, confidence)
    }

    pub fn recognize(&mut self, crop: &DynamicImage) -> Result<(String, f32)> {
        let tensor = self.preprocess(crop);
        let inputs = ort::inputs![TensorRef::from_array_view(&tensor)?];
        let outputs = self.session.run(inputs)?;
        let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let time_steps = shape[1] as usize;
        let vocab_size = shape[2] as usize;
        let logits = logits.to_vec();
        drop(outputs);
        let (text, score) = self.ctc_decode(&logits, time_steps, vocab_size);
        Ok((text, score))
    }
}

fn main() -> Result<()> {
    let det_model_path = "assets/det_int8.onnx";
    let rec_model_path = "assets/rec_int8.onnx";
    let keys_path = "assets/keys.txt";

    let image_paths: Vec<String> = fs::read_dir("../images")?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|s| s.to_str())
                .map_or(false, |ext| matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png"))
        })
        .map(|path| path.to_string_lossy().into_owned())
        .collect();


    let t0 = Instant::now();
    let mut detector = DetectionEngine::new(det_model_path, 640, 640)?;
    let mut recognizer = RecognitionEngine::new(rec_model_path, keys_path)?;
    println!("Model Loading & Initialization Time: {:.2?}", t0.elapsed());
    for image in &image_paths {
        let img = image::open(image).context("Failed to load sample image")?;
        let (orig_w, orig_h) = img.dimensions();
    
        let t_det = Instant::now();
        let boxes = detector.detect(&img)?;
        let det_duration = t_det.elapsed();
    
        let mut ocr_results = Vec::new();
        let t_rec = Instant::now();
        for bbox in &boxes {
            let crop_w = bbox.x_max.saturating_sub(bbox.x_min).min(orig_w - bbox.x_min);
            let crop_h = bbox.y_max.saturating_sub(bbox.y_min).min(orig_h - bbox.y_min);
            if crop_w == 0 || crop_h == 0 {
                continue;
            }
            let cropped_img = img.crop_imm(bbox.x_min, bbox.y_min, crop_w, crop_h);
            let (text, confidence) = recognizer.recognize(&cropped_img)?;
            if !text.is_empty() {
                ocr_results.push(OcrResult {
                    bbox: bbox.clone(),
                    text,
                    confidence,
                });
            }
        }
        let rec_duration = t_rec.elapsed();
    
        println!("Total Latency:           {:.2?}", det_duration + rec_duration);
        println!("Detection Stage Time:    {:.2?}", det_duration);
        println!("Recognition Stage Time:  {:.2?}", rec_duration);
        println!("Average Rec Time/Box:    {:.2?}", rec_duration / boxes.len().max(1) as u32);
        println!("--------------------------------------------------");
    
        for (i, res) in ocr_results.iter().enumerate() {
            println!(
                "Box #{:<2} | Conf: {:.2}% | Text: \"{}\" | BBox: [{}, {}, {}, {}]",
                i + 1,
                res.confidence * 100.0,
                res.text,
                res.bbox.x_min,
                res.bbox.y_min,
                res.bbox.x_max,
                res.bbox.y_max
            );
        }
    }

    Ok(())
}