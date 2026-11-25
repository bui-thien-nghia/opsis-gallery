mod obj_det;
mod ocr;
mod preprocess;
mod profiler;
mod search;
mod storage;
mod types;

use crate::obj_det::YoloDetector;
use crate::ocr::OcrPipeline;
use crate::preprocess::{adaptive_resize, load_image};
use crate::profiler::{StageTimer, SystemSampler, compute_stats, measure_record_storage};
use crate::search::FuzzySearcher;
use crate::storage::{SqliteStore, save_records_json};
use crate::types::{ImageRecord, StageRuntimes};
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use oar_ocr_core::core::inference::set_ort_environment;
use std::collections::HashMap;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

const MAX_DIM: u32 = 1600;
const YOLO_CONF_THRESHOLD: f32 = 0.2;
const OCR_CONF_THRESHOLD: f32 = 0.5;

fn collect_image_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir)
        .with_context(|| format!("Failed to read image directory: {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let ext = path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase());
            if matches!(
                ext.as_deref(),
                Some("jpg" | "jpeg" | "png" | "webp" | "bmp" | "tif" | "tiff")
            ) {
                paths.push(path);
            }
        }
    }
    paths.sort();
    Ok(paths)
}

#[allow(dead_code)]
fn load_lines(path: &Path) -> Result<Vec<String>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("Failed to read label file: {}", path.display()))?;
    Ok(raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(ToOwned::to_owned)
        .collect())
}

#[allow(dead_code)]
fn load_charset(path: &Path) -> Result<Vec<char>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("Failed to read charset file: {}", path.display()))?;
    Ok(raw.chars().filter(|c| *c != '\r' && *c != '\n').collect())
}

fn main() -> Result<()> {
    let env = ort::init()
        .with_telemetry(false)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to initialize ONNX Runtime environment: {e:?}"))?;

    let image_dir = PathBuf::from("D:/test_det");
    let db_path = PathBuf::from("data/gallery.db");
    let json_path = PathBuf::from("data/gallery.json");

    set_ort_environment(env.clone())
        .map_err(|e| anyhow::anyhow!("Failed to share ONNX Runtime environment with OAR: {e:?}"))?;

    let mut yolo = YoloDetector::new(
        &env,
        Path::new("models/obj_det/yolo11n-320-int8.onnx"),
        YOLO_CONF_THRESHOLD,
    )?;
    let ocr = OcrPipeline::new(
        Path::new("models/ocr/ppocrv5-det-mobile.onnx"),
        Path::new("models/ocr/ppocrv5-rec-mobile.onnx"),
        Path::new("models/ocr/ppocrv6-keys.txt"),
        OCR_CONF_THRESHOLD,
    )?;

    let mut sampler = SystemSampler::new()?;
    let image_paths = collect_image_paths(&image_dir)?;

    let pb = ProgressBar::new(image_paths.len() as u64);
    pb.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40} {pos}/{len} {msg} (ETA: {eta_precise})",
    )?);

    let mut records = Vec::with_capacity(image_paths.len());
    let mut proc_times = Vec::with_capacity(image_paths.len());
    let mut proc_mem = Vec::with_capacity(image_paths.len());
    let mut proc_cpu = Vec::with_capacity(image_paths.len());
    let mut storage_totals = Vec::with_capacity(image_paths.len());

    for path in &image_paths {
        let total_timer = StageTimer::start();
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let file_path = path.to_string_lossy().into_owned();

        let mut prep_s = 0.0;
        let mut yolo_detections = Vec::new();
        let mut ocr_detections = Vec::new();
        let mut yolo_s = 0.0;
        let mut ocr_s = 0.0;

        match catch_unwind(AssertUnwindSafe(|| load_image(path))) {
            Ok(Ok(img)) => {
                let prep_timer = StageTimer::start();
                let (resized, scale) = adaptive_resize(img, MAX_DIM);
                let yolo_prep = yolo.prepare(&resized, scale);
                prep_s = prep_timer.elapsed_secs();

                let yolo_timer = StageTimer::start();
                match yolo.detect(yolo_prep) {
                    Ok(detections) => yolo_detections = detections,
                    Err(error) => {
                        eprintln!("YOLO failed for '{}': {error:#}", path.display());
                    }
                }
                yolo_s = yolo_timer.elapsed_secs();

                let ocr_timer = StageTimer::start();
                match ocr.recognize(&resized, scale) {
                    Ok(detections) => ocr_detections = detections,
                    Err(error) => {
                        eprintln!("OCR failed for '{}': {error:#}", path.display());
                    }
                }
                ocr_s = ocr_timer.elapsed_secs();
            }
            Ok(Err(error)) => {
                eprintln!("Unable to process '{}': {error:#}", path.display());
            }
            Err(panic) => {
                let panic_message = panic
                    .downcast_ref::<&str>()
                    .copied()
                    .map(str::to_string)
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                eprintln!(
                    "Unable to process '{}': panic while loading image: {}",
                    path.display(),
                    panic_message
                );
            }
        }

        let post_timer = StageTimer::start();
        let object_classes = yolo_detections
            .iter()
            .map(|d| d.class.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let text_detections = ocr_detections
            .iter()
            .map(|d| d.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let combined_text = if object_classes.is_empty() {
            text_detections.clone()
        } else if text_detections.is_empty() {
            object_classes.clone()
        } else {
            format!("{object_classes} {text_detections}")
        };
        let post_s = post_timer.elapsed_secs();

        let runtime = StageRuntimes {
            preprocessing: prep_s,
            yolo_inference: yolo_s,
            ocr_inference: ocr_s,
            postprocessing: post_s,
            total: total_timer.elapsed_secs(),
        };

        let record = ImageRecord {
            file_name: file_name.clone(),
            file_path,
            object_classes,
            text_detections,
            combined_text,
            yolo_detections,
            ocr_detections,
            runtime,
        };

        let storage_bytes = measure_record_storage(&record);
        let (mem, cpu) = sampler.sample();

        proc_times.push(record.runtime.total);
        proc_mem.push(mem as f64);
        proc_cpu.push(cpu as f64);
        storage_totals.push(storage_bytes.total as f64);
        records.push(record);

        pb.set_message(file_name);
        pb.inc(1);
    }
    pb.finish_with_message("done");

    save_records_json(&records, &json_path)?;

    let insert_timer = StageTimer::start();
    let mut store = SqliteStore::open(&db_path)?;
    store.insert_records(&records)?;
    let insert_elapsed_s = insert_timer.elapsed_secs();
    let (insert_mem, insert_cpu) = sampler.sample();

    let loaded_records = store.load_all_records()?;
    let mut searcher = FuzzySearcher::new();
    let query = "person on a boat";

    let query_timer = StageTimer::start();
    let hits = searcher.search(&loaded_records, query, 10);
    let query_elapsed_s = query_timer.elapsed_secs();
    let (query_mem, query_cpu) = sampler.sample();

    let mut summary: HashMap<&str, _> = HashMap::new();
    summary.insert("per_image_total_s", compute_stats(&proc_times));
    summary.insert("per_image_memory_bytes", compute_stats(&proc_mem));
    summary.insert("per_image_cpu_percent", compute_stats(&proc_cpu));
    summary.insert("per_image_storage_bytes", compute_stats(&storage_totals));

    println!("\n=== Per-Image Processing Stats ===");
    for (metric, stats) in &summary {
        println!(
            "{metric}: min={:.4}, max={:.4}, median={:.4}, mean={:.4}, std_dev={:.4}",
            stats.min, stats.max, stats.median, stats.mean, stats.std_dev
        );
    }

    println!("\n=== Database Insertion ({} records) ===", records.len());
    println!(
        "runtime_s={:.6}, memory_bytes={}, cpu_percent={:.2}",
        insert_elapsed_s, insert_mem, insert_cpu
    );

    println!("\n=== Search Query (\"{}\") ===", query);
    println!(
        "hits={}, runtime_s={:.6}, memory_bytes={}, cpu_percent={:.2}",
        hits.len(),
        query_elapsed_s,
        query_mem,
        query_cpu
    );
    for hit in hits.iter().take(5) {
        println!("  [score={}] {}", hit.score, hit.record.file_name);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[test]
    fn unified_ort_pipeline_runs_yolo_and_ocr_on_image() {
        let env = ort::init().with_telemetry(false).build().unwrap();
        set_ort_environment(env.clone()).unwrap();

        let mut yolo = YoloDetector::new(
            &env,
            Path::new("models/obj_det/yolo11n-320-int8.onnx"),
            YOLO_CONF_THRESHOLD,
        )
        .unwrap();
        let ocr = OcrPipeline::new(
            Path::new("models/ocr/ppocrv5-det-mobile.onnx"),
            Path::new("models/ocr/ppocrv5-rec-mobile.onnx"),
            Path::new("models/ocr/ppocrv6-keys.txt"),
            OCR_CONF_THRESHOLD,
        )
        .unwrap();

        let img = DynamicImage::new_rgb8(640, 480);
        let (resized, scale) = adaptive_resize(img, MAX_DIM);
        let prepared = yolo.prepare(&resized, scale);

        let yolo_out = yolo.detect(prepared).unwrap();
        let ocr_out = ocr.recognize(&resized, scale).unwrap();
        assert!(yolo_out.is_empty());
        assert!(ocr_out.is_empty());
    }
}