mod benchmark;
mod ingest;
mod query;

use crate::benchmark::run_benchmark;
use crate::ingest::{IngestConfig, run_ingest};
use crate::query::run_query_loop;
use anyhow::{Context, Result, bail};
use oar_ocr::core::inference::set_ort_environment;
use std::env;
use std::path::{Path, PathBuf};

const MAX_DIM: u32 = 1600;
const OBJ_CONF_THRESHOLD: f32 = 0.3;
const OCR_CONF_THRESHOLD: f32 = 0.3;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "benchmark".to_string());

    let db_path = PathBuf::from("data/gallery.db");

    match mode.as_str() {
        "ingest" => {
            let dir_arg = args.next().context(
                "Database mode requires an image directory argument: `<exe> db <image_dir>`",
            )?;
            init_shared_ort()?;

            let cfg = IngestConfig {
                image_dir: Path::new(&dir_arg),
                db_path: &db_path,
                obj_model_path: Path::new("models/obj_det/yolo11n-320-fp16.onnx"),
                ocr_det_path: Path::new("models/ocr/ppocrv6-det-s.onnx"),
                ocr_rec_path: Path::new("models/ocr/ppocrv6-rec-s.onnx"),
                ocr_dict_path: Path::new("models/ocr/ppocrv6-keys.txt"),
                max_dim: MAX_DIM,
                obj_conf_threshold: OBJ_CONF_THRESHOLD,
                ocr_conf_threshold: OCR_CONF_THRESHOLD,
            };

            let inserted = run_ingest(&cfg)?;
            println!(
                "Ingested {inserted} image record(s) into '{}'.",
                db_path.display()
            );
        }
        "query" => {
            run_query_loop(&db_path, 10)?;
        }
        "benchmark" => {
            init_shared_ort()?;
            run_benchmark()?;
        }
        other => {
            bail!("Unknown mode '{other}'. Expected one of: db <image_dir>, query, benchmark");
        }
    }

    Ok(())
}

fn init_shared_ort() -> Result<()> {
    let env = ort::init().with_telemetry(false).build()?;
    set_ort_environment(env.clone())
        .map_err(|e| anyhow::anyhow!("Failed to share ONNX Runtime environment with OAR: {e:?}"))?;
    ultralytics_inference::set_ort_environment(env).map_err(|e| {
        anyhow::anyhow!("Failed to share ONNX Runtime environment with Ultralytics: {e:?}")
    })?;
    Ok(())
}
