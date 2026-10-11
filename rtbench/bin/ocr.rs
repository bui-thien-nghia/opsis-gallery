use anyhow::{Context, Result};
use image::{GenericImageView, DynamicImage, ImageBuffer, Rgb, RgbImage};
use ndarray::{Array3, Array4, Axis};
use ort::session::Session;
use ort::value::TensorRef;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

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
        // Initialize ONNX Runtime Session optimized for multi-threaded CPU execution
        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            target_size: (target_w, target_h),
            det_thresh: 0.3,
            box_thresh: 0.5,
        })
    }

    /// Preprocesses image: Resizes to target shape and applies ImageNet normalization
    fn preprocess(&self, img: &DynamicImage) -> (Array4<f32>, f32, f32) {
        let (orig_w, orig_h) = img.dimensions();
        let (target_w, target_h) = self.target_size;

        let scale_w = orig_w as f32 / target_w as f32;
        let scale_h = orig_h as f32 / target_h as f32;

        let resized = img.resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // Standard DBNet ImageNet Normalization
        let mean = [0.485f32, 0.456f32, 0.406f32];
        let std = [0.229f32, 0.224f32, 0.225f32];

        let mut tensor = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));

        for y in 0..target_h as usize {
            for x in 0..target_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let val = pixel[c] as f32 / 255.0;
                    tensor[[0, c, y, x]] = (val - mean[c]) / std[c];
                }
            }
        }

        (tensor, scale_w, scale_h)
    }

    /// Runs detection model and extracts candidate text region bounding boxes
    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<BoundingBox>> {
        let (tensor, scale_w, scale_h) = self.preprocess(img);

        // Run ONNX inference
        let inputs = ort::inputs![TensorRef::from_array_view(&tensor)?]?;
        let outputs = self.session.run(inputs)?;

        // Extract probability map [1, 1, H, W]
        let prob_map = outputs[0].try_extract_tensor::<f32>()?;
        let prob_view = prob_map.view();

        let target_h = self.target_size.1 as usize;
        let target_w = self.target_size.0 as usize;

        // Post-processing: Connected component bounding box extraction
        let mut visited = vec![vec![false; target_w]; target_h];
        let mut boxes = Vec::new();

        for y in 0..target_h {
            for x in 0..target_w {
                let prob = prob_view[[0, 0, y, x]];
                if prob > self.det_thresh && !visited[y][x] {
                    // Flood fill to locate continuous text probability region
                    let mut min_x = x;
                    let mut max_x = x;
                    let mut min_y = y;
                    let mut max_y = y;
                    let mut sum_prob = 0.0f32;
                    let mut count = 0;

                    let mut queue = vec![(x, y)];
                    visited[y][x] = true;

                    while let Some((cx, cy)) = queue.pop() {
                        let p = prob_view[[0, 0, cy, cx]];
                        sum_prob += p;
                        count += 1;

                        min_x = min_x.min(cx);
                        max_x = max_x.max(cx);
                        min_y = min_y.min(cy);
                        max_y = max_y.max(cy);

                        // Check 4-neighbor pixels
                        let neighbors = [
                            (cx.wrapping_sub(1), cy),
                            (cx + 1, cy),
                            (cx, cy.wrapping_sub(1)),
                            (cx, cy + 1),
                        ];

                        for (nx, ny) in neighbors {
                            if nx < target_w && ny < target_h && !visited[ny][nx] {
                                if prob_view[[0, 0, ny, nx]] > self.det_thresh {
                                    visited[ny][nx] = true;
                                    queue.push((nx, ny));
                                }
                            }
                        }
                    }

                    let avg_score = sum_prob / (count as f32);
                    if avg_score >= self.box_thresh && (max_x - min_x > 4) && (max_y - min_y > 4) {
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

// ============================================================================
// 2. Text Recognition Engine (CRNN / SVTR)
// ============================================================================

pub struct RecognitionEngine {
    session: Session,
    vocab: Vec<String>,
    target_height: u32,
    max_width: u32,
}

impl RecognitionEngine {
    pub fn new<P: AsRef<Path>>(model_path: P, keys_path: P) -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        // Read character dictionary (ppocr_keys_v1.txt)
        let file = File::open(keys_path).context("Failed to open dictionary file")?;
        let reader = BufReader::new(file);
        
        // Blank index is token 0 in PP-OCR CTC decoding
        let mut vocab = vec!["<blank>".to_string()];
        for line in reader.lines() {
            vocab.push(line?);
        }
        vocab.push(" ".to_string()); // Append space token

        Ok(Self {
            session,
            vocab,
            target_height: 48,
            max_width: 320,
        })
    }

    /// Resizes crop to target height (48px) maintaining aspect ratio, right-padded to max_width
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
                    // PP-OCR Recognition Normalization: scale to [-1.0, 1.0]
                    tensor[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }

        tensor
    }

    /// Decodes character index sequence using CTC Greedy Search Algorithm
    fn ctc_decode(&self, logits: &[f32], time_steps: usize, vocab_size: usize) -> (String, f32) {
        let mut text = String::new();
        let mut last_idx = 0usize;
        let mut conf_sum = 0.0f32;
        let mut count = 0;

        for t in 0..time_steps {
            let offset = t * vocab_size;
            let step_logits = &logits[offset..offset + vocab_size];

            // Argmax calculation
            let (max_idx, &max_val) = step_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            // Softmax score approximation for confidence measurement
            let exp_sum: f32 = step_logits.iter().map(|v| (v - max_val).exp()).sum();
            let prob = 1.0 / exp_sum;

            // CTC collapse rule: ignore blank token (0) and consecutive duplicates
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

    /// Runs text recognition on a list of cropped image patches
    pub fn recognize(&mut self, crop: &DynamicImage) -> Result<(String, f32)> {
        let tensor = self.preprocess(crop);

        let inputs = ort::inputs![TensorRef::from_array_view(&tensor)?]?;
        let outputs = self.session.run(inputs)?;

        // Shape: [1, TimeSteps, VocabSize]
        let logits_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let shape = logits_tensor.shape();
        let time_steps = shape[1];
        let vocab_size = shape[2];

        let logits_slice = logits_tensor.as_slice().context("Failed to slice logits tensor")?;
        let (text, score) = self.ctc_decode(logits_slice, time_steps, vocab_size);

        Ok((text, score))
    }
}

// ============================================================================
// 3. Complete Benchmark Pipeline
// ============================================================================

fn main() -> Result<()> {
    println!("=== Initializing PaddleOCR INT8 Pipeline in Rust ===");

    let det_model_path = "ppocrv4_det_int8.onnx";
    let rec_model_path = "ppocrv4_rec_int8.onnx";
    let keys_path = "ppocr_keys_v1.txt";
    let sample_image_path = "sample_test.jpg";

    // 1. Initialize Engines
    let t0 = Instant::now();
    let mut detector = DetectionEngine::new(det_model_path, 640, 640)?;
    let mut recognizer = RecognitionEngine::new(rec_model_path, keys_path)?;
    println!("Model Loading & Initialization Time: {:.2?}", t0.elapsed());

    // 2. Load Input Image
    let img = image::open(sample_image_path).context("Failed to load sample image")?;
    let (orig_w, orig_h) = img.dimensions();
    println!("Loaded Input Image: {}x{} px", orig_w, orig_h);

    // 3. Execution: Text Detection
    let t_det = Instant::now();
    let boxes = detector.detect(&img)?;
    let det_duration = t_det.elapsed();
    println!("\n[Stage 1] Detection completed in {:.2?}. Found {} text boxes.", det_duration, boxes.len());

    // 4. Execution: Crop & Recognition Loop
    let mut ocr_results = Vec::new();
    let t_rec = Instant::now();

    for (idx, bbox) in boxes.iter().enumerate() {
        // Crop bounding box region from original image
        let crop_w = bbox.x_max.saturating_sub(bbox.x_min).min(orig_w - bbox.x_min);
        let crop_h = bbox.y_max.saturating_sub(bbox.y_min).min(orig_h - bbox.y_min);

        if crop_w == 0 || crop_h == 0 {
            continue;
        }

        let cropped_img = img.crop_imm(bbox.x_min, bbox.y_min, crop_w, crop_h);
        
        // Run recognition
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

    println!("[Stage 2] Recognition completed in {:.2?} across {} regions.", rec_duration, ocr_results.len());

    // 5. Telemetry & Results Summary
    println!("\n=== Final Benchmark Summary ===");
    println!("Total Latency:           {:.2?}", det_duration + rec_duration);
    println!("Detection Stage Time:    {:.2?}", det_duration);
    println!("Recognition Stage Time:  {:.2?}", rec_duration);
    println!("Average Rec Time/Box:    {:.2?}", rec_duration / (boxes.len().max(1) as u32));
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

    Ok(())
}