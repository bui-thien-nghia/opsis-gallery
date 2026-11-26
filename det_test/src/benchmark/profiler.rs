use crate::ingest::ImageRecord;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use sysinfo::{Pid, ProcessesToUpdate, System};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageRuntimes {
    pub preprocessing: f64,
    pub obj_inference: f64,
    pub ocr_inference: f64,
    pub postprocessing: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct FieldStorageBytes {
    pub file_name: usize,
    pub file_path: usize,
    pub date: usize,
    pub object_classes: usize,
    pub text_detections: usize,
    pub combined_text: usize,
    pub obj_detections: usize,
    pub ocr_detections: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ImageProfile {
    pub file_name: String,
    pub date: String,
    pub runtime_total_s: f64,
    pub runtime_preprocessing_s: f64,
    pub runtime_obj_inference_s: f64,
    pub runtime_ocr_inference_s: f64,
    pub runtime_postprocessing_s: f64,
    pub memory_bytes: u64,
    pub cpu_percent: f32,
    pub storage_bytes: FieldStorageBytes,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct SummaryStats {
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub mean: f64,
    pub std_dev: f64,
}

pub struct StageTimer {
    start: Instant,
}

impl StageTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

pub struct SystemSampler {
    sys: System,
    pid: Option<Pid>,
}

impl SystemSampler {
    pub fn new() -> Result<Self, sysinfo::Error> {
        let pid = sysinfo::get_current_pid().ok();
        let mut sys = System::new()?;
        if let Some(p) = pid {
            sys.refresh_processes(ProcessesToUpdate::Some(&[p]), true);
        }
        Ok(Self { sys, pid })
    }

    pub fn sample(&mut self) -> (u64, f32) {
        let Some(pid) = self.pid else {
            return (0, 0.0);
        };
        self.sys
            .refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        match self.sys.process(pid) {
            Some(proc_) => (proc_.memory(), proc_.cpu_usage()),
            None => (0, 0.0),
        }
    }
}

pub fn measure_record_storage(record: &ImageRecord) -> FieldStorageBytes {
    let bytes_of = |v: &dyn ErasedSerdeLen| v.json_len();
    let file_name = bytes_of(&record.file_name);
    let file_path = bytes_of(&record.file_path);
    let date = bytes_of(&record.date);
    let object_classes = bytes_of(&record.object_classes);
    let text_detections = bytes_of(&record.text_detections);
    let combined_text = bytes_of(&record.combined_text);
    let obj_detections = bytes_of(&record.obj_detections);
    let ocr_detections = bytes_of(&record.ocr_detections);
    let total = file_name
        + file_path
        + date
        + object_classes
        + text_detections
        + combined_text
        + obj_detections
        + ocr_detections;

    FieldStorageBytes {
        file_name,
        file_path,
        date,
        object_classes,
        text_detections,
        combined_text,
        obj_detections,
        ocr_detections,
        total,
    }
}

trait ErasedSerdeLen {
    fn json_len(&self) -> usize;
}

impl<T: Serialize> ErasedSerdeLen for T {
    fn json_len(&self) -> usize {
        serde_json::to_vec(self).map(|b| b.len()).unwrap_or(0)
    }
}

pub fn compute_stats(values: &[f64]) -> SummaryStats {
    if values.is_empty() {
        return SummaryStats::default();
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let min = *sorted.first().unwrap_or(&0.0);
    let max = *sorted.last().unwrap_or(&0.0);
    let n = sorted.len();
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    let sum: f64 = sorted.iter().sum();
    let mean = sum / n as f64;
    let variance: f64 = sorted
        .iter()
        .map(|v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;
    let std_dev = variance.sqrt();

    SummaryStats {
        min,
        max,
        median,
        mean,
        std_dev,
    }
}

pub fn summarize_profiles(profiles: &[ImageProfile]) -> HashMap<String, SummaryStats> {
    let mut metrics: HashMap<&str, Vec<f64>> = HashMap::new();
    for p in profiles {
        metrics
            .entry("runtime_total_s")
            .or_default()
            .push(p.runtime_total_s);
        metrics
            .entry("runtime_preprocessing_s")
            .or_default()
            .push(p.runtime_preprocessing_s);
        metrics
            .entry("runtime_obj_inference_s")
            .or_default()
            .push(p.runtime_obj_inference_s);
        metrics
            .entry("runtime_ocr_inference_s")
            .or_default()
            .push(p.runtime_ocr_inference_s);
        metrics
            .entry("runtime_postprocessing_s")
            .or_default()
            .push(p.runtime_postprocessing_s);
        metrics
            .entry("memory_bytes")
            .or_default()
            .push(p.memory_bytes as f64);
        metrics
            .entry("cpu_percent")
            .or_default()
            .push(p.cpu_percent as f64);
        metrics
            .entry("storage_total_bytes")
            .or_default()
            .push(p.storage_bytes.total as f64);
        metrics
            .entry("storage_file_name")
            .or_default()
            .push(p.storage_bytes.file_name as f64);
        metrics
            .entry("storage_file_path")
            .or_default()
            .push(p.storage_bytes.file_path as f64);
        metrics
            .entry("storage_date")
            .or_default()
            .push(p.storage_bytes.date as f64);
        metrics
            .entry("storage_object_classes")
            .or_default()
            .push(p.storage_bytes.object_classes as f64);
        metrics
            .entry("storage_text_detections")
            .or_default()
            .push(p.storage_bytes.text_detections as f64);
        metrics
            .entry("storage_combined_text")
            .or_default()
            .push(p.storage_bytes.combined_text as f64);
        metrics
            .entry("storage_obj_detections")
            .or_default()
            .push(p.storage_bytes.obj_detections as f64);
        metrics
            .entry("storage_ocr_detections")
            .or_default()
            .push(p.storage_bytes.ocr_detections as f64);
    }

    metrics
        .into_iter()
        .map(|(k, v)| (k.to_string(), compute_stats(&v)))
        .collect()
}
