use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};
use ultralytics_inference::{Device, InferenceConfig, YOLOModel, Quantization};

fn main() -> Result<(), Box<dyn Error>> {
    let quantization = Quantization::Int8;
    let config = InferenceConfig::new()
        .with_max_det(50)
        .with_imgsz(320, 320)
        .with_rect(true)
        .with_confidence(0.25)
        .with_device(Device::Cpu)
        .with_quantize(quantization);
    let mut model = YOLOModel::load_with_config("assets/yoloe-26n-seg_int8.onnx", config)?;

    // COLLECT IMAGE
    let image_paths: Vec<String> = fs::read_dir("images")?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|s| s.to_str())
                .map_or(false, |ext| matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png"))
        })
        .map(|path| path.to_string_lossy().into_owned())
        .collect();

    // PRELOAD IMAGES

    // BENCHMARK RUN
    let mut inference_times_ms = Vec::with_capacity(image_paths.len());
    let pb = ProgressBar::new(image_paths.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{total} ({eta_precise})")
        .unwrap()
        .progress_chars("#>-"));
    for img in &image_paths {
        let start = Instant::now();
        let results = model.predict(img)?;
        let mut unique_class_ids = HashSet::new();
        for result in results {
            if let Some(boxes) = &result.boxes {
                if let Some(ids) = &boxes.id() {
                    for &class_id in ids {
                        unique_class_ids.insert(class_id as usize);
                    }
                }
            }
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        inference_times_ms.push(elapsed_ms);
        pb.inc(1);
    }
    pb.finish_with_message("Done!");
    inference_times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let count = inference_times_ms.len() as f64;
    let min = inference_times_ms.first().cloned().unwrap_or(0.0);
    let max = inference_times_ms.last().cloned().unwrap_or(0.0);
    let sum: f64 = inference_times_ms.iter().sum();
    let avg = sum / count;
    let med = if inference_times_ms.len() % 2 == 0 {
        let mid = inference_times_ms.len() / 2;
        (inference_times_ms[mid - 1] + inference_times_ms[mid]) / 2.0
    } else {
        inference_times_ms[inference_times_ms.len() / 2]
    };
    let variance = inference_times_ms
        .iter()
        .map(|val| (avg - val).powi(2))
        .sum::<f64>() / count;
    let std_dev = variance.sqrt();

    println!("\nInference Time in ms ({} images):", image_paths.len());
    println!("Min: {:.2} ms", min);
    println!("Max: {:.2} ms", max);
    println!("Avg: {:.2} ms", avg);
    println!("Med: {:.2} ms", med);
    println!("Std: {:.2} ms", std_dev);

    Ok(())
}