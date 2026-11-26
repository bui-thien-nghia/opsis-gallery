pub mod profiler;

use crate::ingest::{
    ImageRecord,
    obj_det::ObjDetector,
    ocr::OcrPipeline,
    preprocess::{adaptive_resize, collect_image_paths, formatted_image_date, load_image},
    storage::{SqliteStore, save_records_json},
};
use crate::query::search::FuzzySearcher;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use profiler::{
    ImageProfile, StageRuntimes, StageTimer, SystemSampler, measure_record_storage,
    summarize_profiles,
};
use serde::Serialize;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct BenchmarkCsvRow<'a> {
    row_type: &'a str,
    subject: &'a str,
    date: Option<&'a str>,
    runtime_total_s: Option<f64>,
    runtime_preprocessing_s: Option<f64>,
    runtime_obj_inference_s: Option<f64>,
    runtime_ocr_inference_s: Option<f64>,
    runtime_postprocessing_s: Option<f64>,
    memory_bytes: Option<u64>,
    cpu_percent: Option<f32>,
    storage_file_name_bytes: Option<usize>,
    storage_file_path_bytes: Option<usize>,
    storage_date_bytes: Option<usize>,
    storage_object_classes_bytes: Option<usize>,
    storage_text_detections_bytes: Option<usize>,
    storage_combined_text_bytes: Option<usize>,
    storage_obj_detections_bytes: Option<usize>,
    storage_ocr_detections_bytes: Option<usize>,
    storage_total_bytes: Option<usize>,
    hits: Option<usize>,
    metric: Option<&'a str>,
    min: Option<f64>,
    max: Option<f64>,
    median: Option<f64>,
    mean: Option<f64>,
    std_dev: Option<f64>,
}

impl<'a> BenchmarkCsvRow<'a> {
    fn empty(row_type: &'a str, subject: &'a str) -> Self {
        Self {
            row_type,
            subject,
            date: None,
            runtime_total_s: None,
            runtime_preprocessing_s: None,
            runtime_obj_inference_s: None,
            runtime_ocr_inference_s: None,
            runtime_postprocessing_s: None,
            memory_bytes: None,
            cpu_percent: None,
            storage_file_name_bytes: None,
            storage_file_path_bytes: None,
            storage_date_bytes: None,
            storage_object_classes_bytes: None,
            storage_text_detections_bytes: None,
            storage_combined_text_bytes: None,
            storage_obj_detections_bytes: None,
            storage_ocr_detections_bytes: None,
            storage_total_bytes: None,
            hits: None,
            metric: None,
            min: None,
            max: None,
            median: None,
            mean: None,
            std_dev: None,
        }
    }
}

fn export_benchmark_csv(
    path: &Path,
    profiles: &[ImageProfile],
    summary: &std::collections::HashMap<String, profiler::SummaryStats>,
    insert_runtime_s: f64,
    insert_memory_bytes: u64,
    insert_cpu_percent: f32,
    search_runtime_s: f64,
    search_memory_bytes: u64,
    search_cpu_percent: f32,
    search_hits: usize,
) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;

    for profile in profiles {
        let mut row = BenchmarkCsvRow::empty("image", &profile.file_name);
        row.runtime_total_s = Some(profile.runtime_total_s);
        row.runtime_preprocessing_s = Some(profile.runtime_preprocessing_s);
        row.runtime_obj_inference_s = Some(profile.runtime_obj_inference_s);
        row.runtime_ocr_inference_s = Some(profile.runtime_ocr_inference_s);
        row.runtime_postprocessing_s = Some(profile.runtime_postprocessing_s);
        row.memory_bytes = Some(profile.memory_bytes);
        row.cpu_percent = Some(profile.cpu_percent);
        row.storage_file_name_bytes = Some(profile.storage_bytes.file_name);
        row.storage_file_path_bytes = Some(profile.storage_bytes.file_path);
        row.date = Some(&profile.date);
        row.storage_date_bytes = Some(profile.storage_bytes.date);
        row.storage_object_classes_bytes = Some(profile.storage_bytes.object_classes);
        row.storage_text_detections_bytes = Some(profile.storage_bytes.text_detections);
        row.storage_combined_text_bytes = Some(profile.storage_bytes.combined_text);
        row.storage_obj_detections_bytes = Some(profile.storage_bytes.obj_detections);
        row.storage_ocr_detections_bytes = Some(profile.storage_bytes.ocr_detections);
        row.storage_total_bytes = Some(profile.storage_bytes.total);
        writer.serialize(row)?;
    }

    for (metric, stats) in summary {
        let mut row = BenchmarkCsvRow::empty("summary", metric);
        row.metric = Some(metric);
        row.min = Some(stats.min);
        row.max = Some(stats.max);
        row.median = Some(stats.median);
        row.mean = Some(stats.mean);
        row.std_dev = Some(stats.std_dev);
        writer.serialize(row)?;
    }

    let mut insert_row = BenchmarkCsvRow::empty("operation", "database_insertion");
    insert_row.runtime_total_s = Some(insert_runtime_s);
    insert_row.memory_bytes = Some(insert_memory_bytes);
    insert_row.cpu_percent = Some(insert_cpu_percent);
    writer.serialize(insert_row)?;

    let mut search_row = BenchmarkCsvRow::empty("operation", "search_query");
    search_row.runtime_total_s = Some(search_runtime_s);
    search_row.memory_bytes = Some(search_memory_bytes);
    search_row.cpu_percent = Some(search_cpu_percent);
    search_row.hits = Some(search_hits);
    writer.serialize(search_row)?;

    writer.flush()?;
    Ok(())
}

const MAX_DIM: u32 = 1600;
const OBJ_CONF_THRESHOLD: f32 = 0.3;
const OCR_CONF_THRESHOLD: f32 = 0.3;

pub fn run_benchmark() -> Result<()> {
    let image_dir = PathBuf::from("data/test/samples");
    let db_path = PathBuf::from("data/test/dummy.db");
    let json_path = PathBuf::from("data/test/dataview.json");

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
        let date = formatted_image_date(path);

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

        let combined_text = [
            object_classes.as_str(),
            text_detections.as_str(),
            date.as_str(),
        ]
        .into_iter()
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
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
            date: date.clone(),
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
            date: date.clone(),
            runtime_total_s: runtime.total,
            runtime_preprocessing_s: runtime.preprocessing,
            runtime_obj_inference_s: runtime.obj_inference,
            runtime_ocr_inference_s: runtime.ocr_inference,
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
    let csv_path = PathBuf::from("data/benchmark_profiles.csv");
    export_benchmark_csv(
        &csv_path,
        &profiles,
        &summary,
        insert_elapsed_s,
        insert_mem,
        insert_cpu,
        query_elapsed_s,
        query_mem,
        query_cpu,
        hits.len(),
    )?;
    println!(
        "Benchmark measurements exported to '{}'.",
        csv_path.display()
    );

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
