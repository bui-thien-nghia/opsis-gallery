mod ocr;
mod preprocess;
mod profiler;
mod search;
mod storage;
mod types;
mod obj_det;

use crate::ocr::OcrPipeline;
use crate::preprocess::{adaptive_resize, load_image};
use crate::profiler::{StageTimer, SystemSampler, compute_stats, measure_record_storage};
use crate::search::FuzzySearcher;
use crate::storage::{SqliteStore, save_records_json};
use crate::types::{ImageRecord, StageRuntimes};
use crate::yolo::YoloDetector;
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_DIM: u32 = 1600;
const YOLO_INPUT_SIZE: u32 = 320;
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

fn load_charset(path: &Path) -> Result<Vec<char>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("Failed to read charset file: {}", path.display()))?;
    Ok(raw.chars().filter(|c| *c != '\r' && *c != '\n').collect())
}

fn main() -> Result<()> {
    let image_dir = PathBuf::from("D:/test_img");
    let db_path = PathBuf::from("output/detections.db");
    let json_path = PathBuf::from("output/detections.json");
    let fallback_labels = Path::new("models/obj_det/labels.txt")
        .exists()
        .then(|| load_lines(Path::new("models/obj_det/labels.txt")))
        .transpose()?;
    let charset = load_charset(Path::new("models/ocr/vietnamese_charset.txt"))?;

    let env = ort::init().build()?;
    let mut yolo = YoloDetector::new(
        &env,
        Path::new("models/obj_det/yolo11n_object365.onnx"),
        fallback_labels,
        YOLO_INPUT_SIZE,
        YOLO_CONF_THRESHOLD,
    )?;
    let mut ocr = OcrPipeline::new(
        &env,
        Path::new("models/ocr/craft_mlt_25k_jpqd.onnx"),
        Path::new("models/ocr/latin_g2_jpqd.onnx"),
        charset,
        OCR_CONF_THRESHOLD,
    )?;

    let mut sampler = SystemSampler::new();
    let image_paths = collect_image_paths(&image_dir)?;

    let pb = ProgressBar::new(image_paths.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40} {pos}/{len} {msg}")?,
    );

    let mut records = Vec::with_capacity(image_paths.len());
    let mut proc_times = Vec::with_capacity(image_paths.len());
    let mut proc_mem = Vec::with_capacity(image_paths.len());
    let mut proc_cpu = Vec::with_capacity(image_paths.len());
    let mut storage_totals = Vec::with_capacity(image_paths.len());

    for path in &image_paths {
        // Global timer
        let total_timer = StageTimer::start();

        // Detection preprocessing
        let prep_timer = StageTimer::start();
        let img = load_image(path)?;
        let (resized, scale) = adaptive_resize(img, MAX_DIM);
        let yolo_prep = yolo.prepare(&resized, scale);
        let ocr_det_prep = ocr.prepare_detection(&resized, scale);
        let mut prep_s = prep_timer.elapsed_secs();

        // YOLO inference
        let yolo_timer = StageTimer::start();
        let yolo_detections = yolo.detect(yolo_prep)?;
        let yolo_s = yolo_timer.elapsed_secs();

        // OCR det inference
        let ocr_det_timer = StageTimer::start();
        let (text_boxes, box_scale) = ocr.detect_boxes(ocr_det_prep)?;
        let ocr_det_s = ocr_det_timer.elapsed_secs();

        // Recognition preprocessing
        let ocr_prep_timer = StageTimer::start();
        let ocr_rec_prep = ocr.prepare_recognitions(&resized, &text_boxes, box_scale);
        prep_s += ocr_prep_timer.elapsed_secs();

        // OCR rec inference
        let ocr_rec_timer = StageTimer::start();
        let ocr_detections = ocr.recognize_boxes(ocr_rec_prep)?;
        let ocr_rec_s = ocr_rec_timer.elapsed_secs();

        // Postprocessing
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
        let combined_text = format!("{object_classes} {text_detections}").trim().to_string();

        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let file_path = path.to_string_lossy().into_owned();

        let mut runtime = StageRuntimes {
            preprocessing: prep_s,
            yolo_inference: yolo_s,
            ocr_inference: ocr_det_s + ocr_rec_s,
            postprocessing: 0.0,
            total: 0.0,
        };
        runtime.postprocessing = post_timer.elapsed_secs();

        runtime.total = total_timer.elapsed_secs();

        // Profiling
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
    let query = "invoice receipt table";

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