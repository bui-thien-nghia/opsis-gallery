use crate::ingest::ImageRecord;
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
        let table_exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'detections')",
            [],
            |row| row.get(0),
        )?;

        if !table_exists {
            conn.execute_batch(
                "CREATE TABLE detections (
                file_name TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                object_classes TEXT NOT NULL,
                text_detections TEXT NOT NULL,
                combined_text TEXT NOT NULL,
                obj_detections_json TEXT NOT NULL,
                ocr_detections_json TEXT NOT NULL
            );",
            )?;
        } else {
            let columns = conn
                .prepare("PRAGMA table_info(detections)")?
                .query_map([], |row| row.get::<_, String>(1))?
                .collect::<rusqlite::Result<Vec<_>>>()?;

            if columns
                .iter()
                .any(|column| column == "yolo_detections_json")
            {
                let tx = conn.unchecked_transaction()?;
                tx.execute_batch(
                    "ALTER TABLE detections RENAME TO detections_legacy;
                    CREATE TABLE detections (
                        file_name TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        object_classes TEXT NOT NULL,
                        text_detections TEXT NOT NULL,
                        combined_text TEXT NOT NULL,
                        obj_detections_json TEXT NOT NULL,
                        ocr_detections_json TEXT NOT NULL
                    );
                    INSERT INTO detections (
                        file_name, file_path, object_classes, text_detections, combined_text,
                        obj_detections_json, ocr_detections_json
                    ) SELECT
                        file_name, file_path, object_classes, text_detections, combined_text,
                        yolo_detections_json, ocr_detections_json
                    FROM detections_legacy;
                    DROP TABLE detections_legacy;",
                )?;
                tx.commit()?;
            } else if !columns.iter().any(|column| column == "obj_detections_json") {
                anyhow::bail!("detections table is missing its object-detection JSON column");
            }
        }
        Ok(Self { conn })
    }

    pub fn insert_records(&mut self, records: &[ImageRecord]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO detections (
                    file_name, file_path, object_classes, text_detections, combined_text,
                    obj_detections_json, ocr_detections_json
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;

            for r in records {
                let obj_json = serde_json::to_string(&r.obj_detections)?;
                let ocr_json = serde_json::to_string(&r.ocr_detections)?;
                stmt.execute(params![
                    r.file_name,
                    r.file_path,
                    r.object_classes,
                    r.text_detections,
                    r.combined_text,
                    obj_json,
                    ocr_json,
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
                obj_detections_json, ocr_detections_json
            FROM detections",
        )?;

        let rows = stmt.query_map([], |row| {
            let obj_json: String = row.get(5)?;
            let ocr_json: String = row.get(6)?;
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                obj_json,
                ocr_json,
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
                obj_json,
                ocr_json,
            ) = row?;

            records.push(ImageRecord {
                file_name,
                file_path,
                object_classes,
                text_detections,
                combined_text,
                obj_detections: serde_json::from_str(&obj_json)?,
                ocr_detections: serde_json::from_str(&ocr_json)?,
            });
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::{obj_det::ObjDetection, ocr::OcrDetection};

    fn sample_record() -> ImageRecord {
        ImageRecord {
            file_name: "sample.jpg".to_string(),
            file_path: "images/sample.jpg".to_string(),
            object_classes: "person, boat".to_string(),
            text_detections: "HARBOR".to_string(),
            combined_text: "person, boat HARBOR".to_string(),
            obj_detections: vec![ObjDetection {
                class: "person".to_string(),
                confidence: 0.92,
                bbox: [1.0, 2.0, 30.0, 40.0],
            }],
            ocr_detections: vec![OcrDetection {
                text: "HARBOR".to_string(),
                confidence: 0.88,
                bbox: [[10, 10], [50, 10], [50, 25], [10, 25]],
            }],
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
        assert_eq!(loaded[0].obj_detections.len(), 1);
        assert_eq!(loaded[0].ocr_detections.len(), 1);
        assert_eq!(loaded[0].ocr_detections[0].text, "HARBOR");
    }

    #[test]
    fn sqlite_open_migrates_legacy_yolo_runtime_schema() {
        let dir = std::env::temp_dir().join(format!("det_test_migration_{}", std::process::id()));
        let db_path = dir.join("legacy.db");
        fs::create_dir_all(&dir).unwrap();
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE detections (
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
                );
                INSERT INTO detections VALUES (
                    'legacy.jpg', 'images/legacy.jpg', 'person', '', 'person',
                    '[{\"class\":\"person\",\"confidence\":0.9,\"bbox\":[1.0,2.0,3.0,4.0]}]',
                    '[]', 1.0, 0.1, 0.2, 0.3, 0.4
                );",
            )
            .unwrap();
        }

        let store = SqliteStore::open(&db_path).unwrap();
        let records = store.load_all_records().unwrap();
        drop(store);
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].file_name, "legacy.jpg");
        assert_eq!(records[0].obj_detections.len(), 1);
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
