pub mod obj_det;
pub mod ocr;
pub mod preprocess;
pub mod storage;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use obj_det::ObjDetector;
use ocr::OcrPipeline;
use preprocess::{adaptive_resize, collect_image_paths, load_image};
use serde::{Deserialize, Serialize};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use storage::SqliteStore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRecord {
    pub file_name: String,
    pub file_path: String,
    pub object_classes: String,
    pub text_detections: String,
    pub combined_text: String,
    pub obj_detections: Vec<obj_det::ObjDetection>,
    pub ocr_detections: Vec<ocr::OcrDetection>,
}

pub struct IngestConfig<'a> {
    pub image_dir: &'a Path,
    pub db_path: &'a Path,
    pub obj_model_path: &'a Path,
    pub ocr_det_path: &'a Path,
    pub ocr_rec_path: &'a Path,
    pub ocr_dict_path: &'a Path,
    pub max_dim: u32,
    pub obj_conf_threshold: f32,
    pub ocr_conf_threshold: f32,
}

pub fn run_ingest(cfg: &IngestConfig<'_>) -> Result<usize> {
    let mut obj_detector = ObjDetector::new(cfg.obj_model_path, cfg.obj_conf_threshold)?;
    let ocr = OcrPipeline::new(
        cfg.ocr_det_path,
        cfg.ocr_rec_path,
        cfg.ocr_dict_path,
        cfg.ocr_conf_threshold,
    )?;

    let image_paths = collect_image_paths(cfg.image_dir)?;
    let pb = ProgressBar::new(image_paths.len() as u64);
    pb.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40} {pos}/{len} {msg} (ETA: {eta_precise})",
    )?);

    let mut records = Vec::with_capacity(image_paths.len());

    for path in &image_paths {
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let file_path = path.to_string_lossy().into_owned();

        let mut obj_detections = Vec::new();
        let mut ocr_detections = Vec::new();

        match catch_unwind(AssertUnwindSafe(|| load_image(path))) {
            Ok(Ok(img)) => {
                let (resized, scale) = adaptive_resize(img, cfg.max_dim);
                let obj_input = obj_detector.prepare(&resized, scale);

                if let Err(error) = obj_detector.detect(obj_input).map(|d| obj_detections = d) {
                    eprintln!(
                        "Object detection failed for '{}': {error:#}",
                        path.display()
                    );
                }

                if let Err(error) = ocr.recognize(&resized, scale).map(|d| ocr_detections = d) {
                    eprintln!("OCR failed for '{}': {error:#}", path.display());
                }
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

        records.push(ImageRecord {
            file_name: file_name.clone(),
            file_path,
            object_classes,
            text_detections,
            combined_text,
            obj_detections,
            ocr_detections,
        });

        pb.set_message(file_name);
        pb.inc(1);
    }
    pb.finish_with_message("done");

    let count = records.len();
    let mut store = SqliteStore::open(cfg.db_path)?;
    store.insert_records(&records)?;
    Ok(count)
}
