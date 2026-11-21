use crate::types::ImageRecord;
use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::fs;
use std::path::Path;

pub fn save_json(records: &[ImageRecord], output_path: &Path) -> Result<()> {
    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(records)?;
    fs::write(output_path, json)
        .with_context(|| format!("Failed to write JSON output to {}", output_path.display()))?;
    Ok(())
}

pub struct SqliteStore {
    conn: Connection,
}

impl SqliteStore {
    pub fn new(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open SQLite database at {}", db_path.display()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS detections (
                file_name TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                object_classes TEXT NOT NULL,
                text_detections TEXT NOT NULL,
                combined_text TEXT NOT NULL,
                yolo_detections_json TEXT NOT NULL,
                ocr_detections_json TEXT NOT NULL,
                runtime_json TEXT NOT NULL
            );",
        )?;
        Ok(Self { conn })
    }

    pub fn insert_records(&mut self, records: &[ImageRecord]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO detections (
                    file_name,
                    file_path,
                    object_classes,
                    text_detections,
                    combined_text,
                    yolo_detections_json,
                    ocr_detections_json,
                    runtime_json
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;

            for rec in records {
                let yolo_json = serde_json::to_string(&rec.yolo_detections)?;
                let ocr_json = serde_json::to_string(&rec.ocr_detections)?;
                let runtime_json = serde_json::to_string(&rec.runtime)?;

                stmt.execute(params![
                    rec.file_name,
                    rec.file_path,
                    rec.object_classes,
                    rec.text_detections,
                    rec.combined_text,
                    yolo_json,
                    ocr_json,
                    runtime_json,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn load_all_records(&self) -> Result<Vec<ImageRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT
                file_name,
                file_path,
                object_classes,
                text_detections,
                combined_text,
                yolo_detections_json,
                ocr_detections_json,
                runtime_json
            FROM detections",
        )?;

        let rows = stmt.query_map([], |row| {
            let file_name: String = row.get(0)?;
            let file_path: String = row.get(1)?;
            let object_classes: String = row.get(2)?;
            let text_detections: String = row.get(3)?;
            let combined_text: String = row.get(4)?;
            let yolo_json: String = row.get(5)?;
            let ocr_json: String = row.get(6)?;
            let runtime_json: String = row.get(7)?;

            Ok((
                file_name,
                file_path,
                object_classes,
                text_detections,
                combined_text,
                yolo_json,
                ocr_json,
                runtime_json,
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
                runtime_json,
            ) = row?;

            records.push(ImageRecord {
                file_name,
                file_path,
                object_classes,
                text_detections,
                combined_text,
                yolo_detections: serde_json::from_str(&yolo_json)?,
                ocr_detections: serde_json::from_str(&ocr_json)?,
                runtime: serde_json::from_str(&runtime_json)?,
            });
        }

        Ok(records)
    }
}