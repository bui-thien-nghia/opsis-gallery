use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloDetection {
    pub class: String,
    pub confidence: f32,
    pub bbox: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrDetection {
    pub text: String,
    pub confidence: f32,
    pub bbox: [[i32; 2]; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageRuntimes {
    pub preprocessing: f64,
    pub yolo_inference: f64,
    pub ocr_inference: f64,
    pub postprocessing: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRecord {
    pub file_name: String,
    pub file_path: String,
    pub object_classes: String,
    pub text_detections: String,
    pub combined_text: String,
    pub yolo_detections: Vec<YoloDetection>,
    pub ocr_detections: Vec<OcrDetection>,
    pub runtime: StageRuntimes,
}