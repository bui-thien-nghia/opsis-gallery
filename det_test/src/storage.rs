use crate::types::ImageRecord;
use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::fs;
use std::path::Path;

pub fn save_records_json(records: &[ImageRecord], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(records)?;
    fs::write(path, json).with_context(|| format!("Failed to write JSON to {}", path.display()))
}

pub struct SqliteStore {
    conn: Connection,
}

impl SqliteStore {
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open SQLite DB at {}", db_path.display()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS detections (
                file_name TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                object_classes TEXT NOT NULL,
                text_detections TEXT NOT NULL,
                combined_text TEXT NOT NULL,
                yolo_detections_json TEXT NOT NULL,
                ocr_detections_json TEXT NOT NULL,
                runtime_total_s REAL NOT NULL,
                runtime_preprocessing_s REAL NOT NULL,
                runtime_yolo_inference_s REAL NOT NULL,
                runtime_ocr_inference_s REAL NOT NULL,
                runtime_postprocessing_s REAL NOT NULL
            );",
        )?;
        Ok(Self { conn })
    }

    pub fn insert_records(&mut self, records: &[ImageRecord]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO detections (
                    file_name, file_path, object_classes, text_detections, combined_text,
                    yolo_detections_json, ocr_detections_json,
                    runtime_total_s, runtime_preprocessing_s, runtime_yolo_inference_s,
                    runtime_ocr_inference_s, runtime_postprocessing_s
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            )?;

            for r in records {
                let yolo_json = serde_json::to_string(&r.yolo_detections)?;
                let ocr_json = serde_json::to_string(&r.ocr_detections)?;
                stmt.execute(params![
                    r.file_name,
                    r.file_path,
                    r.object_classes,
                    r.text_detections,
                    r.combined_text,
                    yolo_json,
                    ocr_json,
                    r.runtime.total,
                    r.runtime.preprocessing,
                    r.runtime.yolo_inference,
                    r.runtime.ocr_inference,
                    r.runtime.postprocessing,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn load_all_records(&self) -> Result<Vec<ImageRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT
                file_name, file_path, object_classes, text_detections, combined_text,
                yolo_detections_json, ocr_detections_json,
                runtime_total_s, runtime_preprocessing_s, runtime_yolo_inference_s,
                runtime_ocr_inference_s, runtime_postprocessing_s
            FROM detections",
        )?;

        let rows = stmt.query_map([], |row| {
            let yolo_json: String = row.get(5)?;
            let ocr_json: String = row.get(6)?;
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                yolo_json,
                ocr_json,
                row.get::<_, f64>(7)?,
                row.get::<_, f64>(8)?,
                row.get::<_, f64>(9)?,
                row.get::<_, f64>(10)?,
                row.get::<_, f64>(11)?,
            ))
        })?;

        let mut records = Vec::new();
        for row in rows {
            let (
                file_name,
                file_path,
                object_classes,
                text_detections,
                combined_text,
                yolo_json,
                ocr_json,
                total,
                preprocessing,
                yolo_inference,
                ocr_inference,
                postprocessing,
            ) = row?;

            records.push(ImageRecord {
                file_name,
                file_path,
                object_classes,
                text_detections,
                combined_text,
                yolo_detections: serde_json::from_str(&yolo_json)?,
                ocr_detections: serde_json::from_str(&ocr_json)?,
                runtime: crate::types::StageRuntimes {
                    preprocessing,
                    yolo_inference,
                    ocr_inference,
                    postprocessing,
                    total,
                },
            });
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OcrDetection, StageRuntimes, YoloDetection};

    fn sample_record() -> ImageRecord {
        ImageRecord {
            file_name: "sample.jpg".to_string(),
            file_path: "images/sample.jpg".to_string(),
            object_classes: "person, boat".to_string(),
            text_detections: "HARBOR".to_string(),
            combined_text: "person, boat HARBOR".to_string(),
            yolo_detections: vec![YoloDetection {
                class: "person".to_string(),
                confidence: 0.92,
                bbox: [1.0, 2.0, 30.0, 40.0],
            }],
            ocr_detections: vec![OcrDetection {
                text: "HARBOR".to_string(),
                confidence: 0.88,
                bbox: [[10, 10], [50, 10], [50, 25], [10, 25]],
            }],
            runtime: StageRuntimes {
                preprocessing: 0.01,
                yolo_inference: 0.02,
                ocr_inference: 0.03,
                postprocessing: 0.005,
                total: 0.065,
            },
        }
    }

    #[test]
    fn sqlite_round_trip_preserves_records() {
        let dir = std::env::temp_dir().join(format!("det_test_storage_{}", std::process::id()));
        let db_path = dir.join("test.db");
        let record = sample_record();

        let mut store = SqliteStore::open(&db_path).unwrap();
        store.insert_records(std::slice::from_ref(&record)).unwrap();
        let loaded = store.load_all_records().unwrap();
        drop(store);
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].file_name, record.file_name);
        assert_eq!(loaded[0].combined_text, record.combined_text);
        assert_eq!(loaded[0].yolo_detections.len(), 1);
        assert_eq!(loaded[0].ocr_detections.len(), 1);
        assert_eq!(loaded[0].ocr_detections[0].text, "HARBOR");
    }

    #[test]
    fn save_records_json_writes_valid_array() {
        let dir = std::env::temp_dir().join(format!("det_test_json_{}", std::process::id()));
        let json_path = dir.join("records.json");
        let record = sample_record();

        save_records_json(std::slice::from_ref(&record), &json_path).unwrap();
        let raw = fs::read_to_string(&json_path).unwrap();
        let _ = fs::remove_dir_all(&dir);

        let parsed: Vec<ImageRecord> = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].file_name, "sample.jpg");
    }
}
