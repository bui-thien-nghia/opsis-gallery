pub mod profiler;

use crate::ingest::{
    ImageRecord,
    obj_det::ObjDetector,
    ocr::OcrPipeline,
    preprocess::{adaptive_resize, collect_image_paths, load_image},
    storage::{SqliteStore, save_records_json},
};
use crate::query::search::FuzzySearcher;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use profiler::{
    ImageProfile, StageRuntimes, StageTimer, SystemSampler, measure_record_storage,
    summarize_profiles,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

const MAX_DIM: u32 = 1600;
const OBJ_CONF_THRESHOLD: f32 = 0.3;
const OCR_CONF_THRESHOLD: f32 = 0.3;

pub fn run_benchmark() -> Result<()> {
    let image_dir = PathBuf::from("D:/test_det");
    let db_path = PathBuf::from("data/gallery.db");
    let json_path = PathBuf::from("data/gallery.json");

    let mut obj_detector = ObjDetector::new(
        Path::new("models/obj_det/yolo11n-320-fp16.onnx"),
        OBJ_CONF_THRESHOLD,
    )?;
    let ocr = OcrPipeline::new(
        Path::new("models/ocr/ppocrv6-det-s.onnx"),
        Path::new("models/ocr/ppocrv6-rec-s.onnx"),
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
    let mut profiles = Vec::with_capacity(image_paths.len());

    for path in &image_paths {
        let total_timer = StageTimer::start();
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let file_path = path.to_string_lossy().into_owned();

        let mut prep_s = 0.0;
        let mut obj_detections = Vec::new();
        let mut ocr_detections = Vec::new();
        let mut obj_s = 0.0;
        let mut ocr_s = 0.0;

        match catch_unwind(AssertUnwindSafe(|| load_image(path))) {
            Ok(Ok(img)) => {
                let prep_timer = StageTimer::start();
                let (resized, scale) = adaptive_resize(img, MAX_DIM);
                let obj_input = obj_detector.prepare(&resized, scale);
                prep_s = prep_timer.elapsed_secs();

                let obj_timer = StageTimer::start();
                match obj_detector.detect(obj_input) {
                    Ok(detections) => obj_detections = detections,
                    Err(error) => {
                        eprintln!(
                            "Object detection failed for '{}': {error:#}",
                            path.display()
                        );
                    }
                }
                obj_s = obj_timer.elapsed_secs();

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
        let object_classes = obj_detections
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
            obj_inference: obj_s,
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
            obj_detections,
            ocr_detections,
        };

        let storage_bytes = measure_record_storage(&record);
        let (mem, cpu) = sampler.sample();

        profiles.push(ImageProfile {
            file_name: file_name.clone(),
            runtime_total_s: runtime.total,
            runtime_preprocessing_s: runtime.preprocessing,
            runtime_inference_s: runtime.obj_inference + runtime.ocr_inference,
            runtime_postprocessing_s: runtime.postprocessing,
            memory_bytes: mem,
            cpu_percent: cpu,
            storage_bytes,
        });
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
    let query = "document";

    let query_timer = StageTimer::start();
    let hits = searcher.search(&loaded_records, query, 10);
    let query_elapsed_s = query_timer.elapsed_secs();
    let (query_mem, query_cpu) = sampler.sample();

    let summary = summarize_profiles(&profiles);

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
