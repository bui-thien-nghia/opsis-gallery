# Opsis Gallery Repository Guide

## Overview

Opsis Gallery is a Rust and ONNX-focused computer vision workspace containing two independent benchmark-oriented programs:

- `det_test` builds an image gallery search and benchmark pipeline from object detection, OCR, metadata, SQLite storage, fuzzy text search, JSON and CSV export, and runtime monitoring.
- `siglip2_test` benchmarks text and vision embedding models with ONNX Runtime and reports memory and CPU telemetry.

The repository is organized as a collection of standalone Rust crates rather than as a single Cargo workspace. Run each crate from its own directory. The two programs are intended to be integrated into a larger application or used as independent experimental components.

The repository also contains an optional Python virtual environment and notebook support files. They are not required to compile or run the Rust projects in this guide.

## Repository Layout

```text
Opsis Gallery/
  det_test/
    data/
      test/
        samples/
    models/
      obj_det/
      ocr/
    patches/
      ultralytics-inference/
    src/
      main.rs
      benchmark.rs
      benchmark/profiler.rs
      ingest.rs
      ingest/obj_det.rs
      ingest/ocr.rs
      ingest/preprocess.rs
      ingest/storage.rs
      query.rs
      query/search.rs
    Cargo.toml
  siglip2_test/
    benches/
    models/
    src/
      main.rs
      enc.rs
      enc/dataset.rs
      enc/text.rs
      enc/vision.rs
      enc/telemetry.rs
    Cargo.toml
  requirements.txt
  monitoring/
```

## Prerequisites

### Required Software

For both projects, Rust 2024 edition with Cargo is required, using a recent stable toolchain.

A Windows command prompt or PowerShell session should be used from the repository root or from each crate directory.

Check Rust:

```powershell
rustc --version
cargo --version
```

### Model Assets

The models used for both projects can be found in [this link](https://drive.google.com/drive/folders/1_Vv3uig5-gc3d7oH3TIxZaL8W4ADnO8f?usp=sharing). To use the models, put them inside the allocated folders inside the `models/` folder under the corresponding project folder.

Currently, `det_test` uses:

- `det_test/models/obj_det/yolo11n-320-fp16.onnx` for object detection.
- `det_test/models/ocr/ppocrv6-det-s.onnx` for text-region detection.
- `det_test/models/ocr/ppocrv6-rec-s.onnx` for text recognition.
- `det_test/models/ocr/ppocrv6-keys.txt` for the OCR character dictionary.

The alternative ONNX files in `models/obj_det/` and `models/ocr/` can be used only after changing the model paths and confirming compatibility with the configured engine options.

Currently, `siglip2_test` uses:

- `siglip2_test/models/vision.onnx`
- `siglip2_test/models/text.onnx`
- `siglip2_test/models/tokenizer.json`

## Building the Repository

Build and check each Rust crate separately.

### `det_test`

```powershell
cd det_test
cargo check
cargo build --release
```

Run the release executable:

```powershell
cargo run --release -- <mode> [image-directory]
```

### `siglip2_test`

```powershell
cd siglip2_test
cargo check
cargo build --release
```

Run the benchmark binary:

```powershell
cargo run --release -- --image-dir D:/test_det
```

A model or tokenizer command-line argument can be overridden when running the binary:

```powershell
cargo run --release -- --image-dir D:/test_det --vision-model models/vision.onnx --text-model models/text.onnx --tokenizer models/tokenizer.json --threads 4
```

## `det_test`

### Purpose

`det_test` combines:

- Ultralytics-based object detection.
- OAR-OCR text detection and recognition.
- EXIF image-date extraction.
- SQLite record storage.
- JSON record export.
- Fuzzy record search.
- Per-image, per-operation, and storage profiling.
- CSV benchmark export.

The implementation is divided into three main modules:

- `ingest`: inference and database insertion.
- `query`: interactive fuzzy-search mode.
- `benchmark`: the profile and performance harness.

The detector and OCR implementations share one ONNX Runtime environment. This avoids starting independent ONNX Runtime environments for the object and OCR pipelines.

### Operating Modes

The executable has three modes.

#### 1. Ingest Mode

Ingest an image directory and insert the resulting records into `data/gallery.db`.

```powershell
cd det_test
cargo run --release -- ingest D:/test_det
```

The release executable form is:

```powershell
.\target\release\det_test.exe ingest D:/test_det
```

In this mode:

1. Images are collected from the supplied directory.
2. Images are decoded and resized without enlarging an already-small image.
3. Object detection produces `ObjDetection` values.
4. OCR produces `OcrDetection` values.
5. The image capture date is extracted from EXIF metadata when available.
6. Object names, OCR text, and formatted date are joined into `combined_text`.
7. Records are inserted into SQLite and assigned the configured primary file name.

The ingestion configuration is in `src/main.rs`. Update `IngestConfig` values if an application must select different models, directories, confidence thresholds, or maximum image dimensions.

Important behavior:

- Invalid or unreadable image files are logged and skipped instead of stopping the entire ingestion run.
- Panics caused during image loading are caught and reported per file.
- The maximum image side is currently `1600` pixels.
- The default object-confidence threshold is `0.3`.
- The default OCR-confidence threshold is `0.3`.
- Only an object-detection instance and an OCR instance are loaded for the run.
- The detector and recognizer keep one process-wide ONNX environment in the shared setup used by `main.rs`.

#### 2. Query Mode

Open the records database and enter an interactive fuzzy-search loop.

```powershell
cd det_test
cargo run --release -- query
```

Search results are printed by file path. Enter a blank line once, and then a second blank line to exit. Submitting a query that is not empty resets the blank-line counter.

The database is fixed in query mode to:

```text
data/gallery.db
```

The query engine searches `combined_text`. Therefore, every combined value should include:

- Detected object classes.
- Recognized OCR text.
- The formatted image date when EXIF metadata is available.

Example:

```text
query> person boat October 2026
query> document
query>
query>
```

A blank line followed immediately by another blank line exits the loop.

#### 3. Benchmark Mode

Run the benchmark harness:

```powershell
cd det_test
cargo run --release -- benchmark
```

No benchmark arguments are currently read from the command line. The benchmark paths are fixed in `src/benchmark.rs`:

```text
Image directory: data/test/samples
Database:        data/test/dummy.db
JSON export:     data/test/dataview.json
CSV export:      data/benchmark_profiles.csv
```

Adjust these paths and model selections in `run_benchmark()` before benchmarking another dataset or different model combination.

Benchmark mode:

- Runs image preprocessing.
- Measures object detection.
- Measures OCR recognition.
- Measures post-processing.
- Measures total per-image processing time.
- Samples process RSS memory and CPU usage.
- Measures JSON storage size for each record field.
- Inserts all records into SQLite.
- Executes a fixed search query.
- Prints summary statistics.
- Exports the tabular benchmark measurements to CSV.

### Image Date Extraction

`ingest/preprocess.rs` provides `formatted_image_date()`. It reads image EXIF metadata through `kamadak-exif` and checks date fields in this order:

1. `DateTimeOriginal`
2. `DateTimeDigitized`
3. `DateTime`

The output format is:

```text
[Written month] [Ordered day], [Year]
```

Examples:

```text
October 20th, 2026
January 1st, 2026
February 2nd, 2026
March 3rd, 2026
April 11th, 2026
```

Supported date representations include normal EXIF colon-separated dates and ISO-style dash-separated dates. Invalid calendar dates, including invalid leap-day values, are rejected.

If no date metadata can be read, `date` is an empty string. The current fallback policy does not derive a date from the file name or filesystem modification time. Applications that require a date for every record should add an explicit fallback policy before this function is reused.

### `ImageRecord`

The central record is `ingest::ImageRecord`. It contains:

```rust
pub struct ImageRecord {
    pub file_name: String,
    pub file_path: String,
    pub date: String,
    pub object_classes: String,
    pub text_detections: String,
    pub combined_text: String,
    pub obj_detections: Vec<obj_det::ObjDetection>,
    pub ocr_detections: Vec<ocr::OcrDetection>,
}
```

Important integration points:

- `obj_detections` contains the structured object-detection output.
- `ocr_detections` contains structured OCR text and quadrilateral coordinates.
- `combined_text` is the field used for fuzzy searching.
- `date` is part of `combined_text` and of the benchmark storage measurement.
- JSON export serializes the complete record structure.
- SQLite stores detection arrays as JSON text inside the `detections` table.

An image date field deserializes as an empty string when loading older JSON records that do not contain `date`.

### SQLite and JSON Output

New SQLite databases use a `detections` table containing:

- `file_name`
- `file_path`
- `date`
- `object_classes`
- `text_detections`
- `combined_text`
- `obj_detections_json`
- `ocr_detections_json`

`SqliteStore::open()` performs compatibility migration for legacy schemas.

Legacy databases that contain `yolo_detections_json` are migrated to `obj_detections_json`. An existing table that has no `date` column receives `date TEXT NOT NULL DEFAULT ''`. Runtime columns from older database versions are not preserved in `ImageRecord`; benchmark runtime data is handled separately by `ImageProfile` and CSV export.

Database output:

```text
data/gallery.db
```

Ingest JSON export is not run in `run_ingest()`. JSON export is currently used by benchmark mode and defaults to:

```text
data/test/dataview.json
```

Database insertions use `INSERT OR REPLACE`, with `file_name` as the primary key. Ingesting a file with the same name replaces its existing database row.

### Benchmark CSV Output

Benchmark mode writes a normalized wide-format CSV file to:

```text
data/benchmark_profiles.csv
```

The CSV contains three row categories:

- `image` rows for each processed file.
- `summary` rows for aggregate min, max, median, mean, and standard deviation measurements.
- `operation` rows for database insertion and search query operations.

Per-image columns include:

- Image file name and date.
- Total runtime.
- Preprocessing runtime.
- Object-detection runtime.
- OCR runtime.
- Post-processing runtime.
- RSS memory.
- CPU usage.
- Storage size for file name, path, date, object classes, OCR text, combined text, object detections, OCR detections, and total record size.

The CSV is intended for downstream spreadsheet analysis, plotting, or integration with monitoring tools. It is not a replacement for the SQLite or JSON record formats.

### Detection and OCR Notes

The object detector wraps `ultralytics-inference`. The OCR implementation wraps `oar-ocr`.

For an application that reuses both engines in the same process:

1. Create the shared ONNX environment once with `ort::init().build()`.
2. Install it in OAR-OCR with `oar_ocr::core::inference::set_ort_environment(env.clone())`.
3. Install the same environment in Ultralytics with `ultralytics_inference::set_ort_environment(env)`.
4. Only then construct detector and OCR models.

Do not initialize a second ONNX environment after any model or session has started. Also avoid constructing test environments in parallel if tests share a process-wide ONNX Runtime environment.

The current ONNX environment setup is implemented in `main.rs` and may need to be called before each long-lived application mode, but must not be called more than once for the same process.

### Image Loading Notes

`load_image()` uses format detection rather than filename extension alone. This allows some valid images with a mismatched extension to decode when the underlying format is recognized.

Supported collection extensions include JPEG, PNG, WebP, BMP, TIFF, GIF, and HEIC/XMP variants in the current source. HEIC support is format-dependent and is not guaranteed unless the image crate and platform decoder provide the required format. An application that must process HEIC or RAW-derived image families should use an explicit decoder or convert the files first.

The benchmark and ingest modules catch image decode errors and continue. A record may be inserted with empty detections if loading or inference fails.

## `siglip2_test`

### Purpose

`siglip2_test`, internally named `rtbench`, measures CPU inference and memory behavior for SigLIP2-style or compatible text and vision ONNX encoders.

It provides:

- Text prompt tokenization and embedding inference.
- Vision image preprocessing and embedding inference.
- RSS memory sampling.
- CPU utilization sampling.
- Throughput calculations.
- A periodic CSV memory timeline.
- Criterion benchmarks for preprocessing and inference.

The library modules are located in `src/enc.rs` and `src/enc/`.

### Required Assets

The default runtime assets are:

```text
siglip2_test/models/vision.onnx
siglip2_test/models/text.onnx
siglip2_test/models/tokenizer.json
```

If a model or tokenizer is missing, the corresponding benchmark stage is skipped instead of terminating the entire run. A missing vision model skips the image pipeline. A missing text model or tokenizer skips the 1000-prompt text benchmark.

An integrating application should validate asset paths before constructing `TextEngine` or `VisionEngine`.

### Running the Runner

From `siglip2_test/`:

```powershell
cargo run --release -- --image-dir D:/test_det --threads 4
```

The text prompt can be customized:

```powershell
cargo run --release -- --image-dir D:/test_det --prompt "A document on a desk" --threads 4
```

The complete command-line options are:

```text
--vision-model <PATH>  Vision ONNX model; default: models/vision.onnx
--text-model <PATH>    Text ONNX model; default: models/text.onnx
--tokenizer <PATH>     Hugging Face tokenizer JSON; default: models/tokenizer.json
--image-dir <PATH>     Directory of benchmark images; required
--prompt <TEXT>        Text prompt; default text is a grill-party description
--threads <COUNT>      ONNX intra-op thread count; default: 2
```

For example:

```powershell
cargo run --release -- --image-dir D:/test_det --vision-model models/vision.onnx --text-model models/text.onnx --tokenizer models/tokenizer.json --prompt "A person holding a document" --threads 2
```

### Text Benchmark

When the text model and tokenizer exist, the runner:

1. Loads the tokenizer.
2. Loads the ONNX text model.
3. Tokenizes the configured prompt.
4. Runs the prompt 1000 times.
5. Measures elapsed inference time.
6. Samples RSS before and after text inference.

The text engine:

- Expects the model input name `input_ids`.
- Expects an output named `pooler_output`.
- Converts the output vector to a single embedding vector.
- L2-normalizes the embedding.

The tokenizer must produce IDs compatible with the exported ONNX model. Mismatched sequence length, padding behavior, or vocabulary is a model compatibility issue rather than a Rust dataset issue.

### Vision Benchmark

When the vision model exists and the image directory contains valid images, the runner:

1. Loads the ONNX vision model.
2. Scans the selected directory for `.jpg`, `.jpeg`, or `.png` files.
3. Sorts paths.
4. Resizes each image to `224 x 224`.
5. Converts pixels to a normalized CHW tensor.
6. Runs inference.
7. Reads `pooler_output`.
8. L2-normalizes the resulting embedding.

Preprocessing and inference times are accumulated separately. The current dataset module is a flat-directory iterator and does not recursively walk subdirectories.

The preprocessing formula is:

```text
pixel / 127.5 - 1.0
```

The input tensor shape is:

```text
(1, 3, 224, 224)
```

An ONNX model with different input names, dynamic shapes, preprocessing values, or output names must be adapted in `src/enc/vision.rs` or `src/enc/text.rs`.

### Memory and CPU Telemetry

A background telemetry thread starts before text benchmarking and samples:

- CPU usage.
- RSS memory.
- Elapsed runtime.

The runner prints:

- Peak CPU utilization.
- Peak RSS memory.
- Text RSS after model creation.
- Text RSS after inference.
- Vision RSS after model creation.
- Vision RSS after inference.
- RSS immediately before shutdown.

Telemetry samples are written to:

```text
siglip2_test/memory_trace.csv
```

The CSV columns are:

```text
elapsed_ms,elapsed_secs,rss_bytes,rss_mb,cpu_usage_pct
```

If telemetry writing fails, it is reported to standard error, but the benchmark can continue or finish.

### Criterion Benchmarks

The `siglip2_test/benches/bench.rs` source defines Criterion groups for:

- Vision image preprocessing at small, medium, and large sample sizes.
- Vision inference with 1, 2, and 4 threads.
- Text inference with 1, 2, and 4 threads.

The benchmark source currently assumes test assets at:

```text
assets/images/
assets/enc/vision.onnx
assets/enc/text.onnx
assets/enc/tokenizer.json
```

The repository currently supplies the models under `models/` rather than `assets/enc/`, so update the hard-coded paths or copy assets to the path layout expected by `bench.rs` before benchmarking.

The Criterion benchmark target imports `rtbench::enc`. The package has `src/enc.rs` but no `src/lib.rs` in the supplied layout. An integrating application that needs Criterion should add a library target exporting `enc`, or change the bench target to compile the modules through the package structure it requires.

Run only after those benchmark-specific paths and library integration are aligned:

```powershell
cd siglip2_test
cargo bench --bench bench
```

Criterion output is managed by Criterion and normally appears in `target/criterion/`.

## Integration Guidance

`det_test` is the project intended for integration into the Opsis Gallery application or another gallery workflow. `siglip2_test` remains an independent research and benchmarking project. It demonstrates text and vision embedding behavior, but it is not required for the `det_test` search and ingestion pipeline.

### Integrating `det_test`

A host application should treat `det_test` as a data and inference component rather than as a finished UI or service. The repository currently exposes command-oriented entry points and fixed paths, so integration normally begins with the following adaptations:

1. Move model paths, database paths, JSON paths, benchmark paths, image directories, and confidence thresholds into application configuration.
2. Wrap `run_ingest()` in a library or service API that reports progress and accepts an explicit cancellation or error policy.
3. Return `ImageRecord` values instead of relying on terminal progress output.
4. Choose SQLite, JSON, or an application-defined storage backend as the authoritative record source.
5. Preserve `combined_text` for fuzzy search while exposing structured detection and date fields to the UI.
6. Serialize detection coordinates consistently if results are sent outside the Rust process.
7. Keep benchmark profiling outside the production ingest path.

The recommended integration sequence is:

```text
Image directory
  -> Image loading and format detection
  -> EXIF date extraction
  -> Object detection
  -> OCR detection and recognition
  -> ImageRecord construction
  -> SQLite/JSON/storage write
  -> Combined text indexing
  -> Fuzzy search
```

Before indexing, construct `combined_text` from the detected object names, recognized text, and formatted image date. The fuzzy-search implementation in `query/search.rs` currently searches this field only. Structured fields such as `obj_detections`, `ocr_detections`, and `date` remain available for filtering or rendering without being added directly to the search text.

#### Application-facing record format

The primary integration contract is `ingest::ImageRecord`:

```rust
pub struct ImageRecord {
    pub file_name: String,
    pub file_path: String,
    pub date: String,
    pub object_classes: String,
    pub text_detections: String,
    pub combined_text: String,
    pub obj_detections: Vec<obj_det::ObjDetection>,
    pub ocr_detections: Vec<ocr::OcrDetection>,
}
```

Where `ObjDetection` and `OcrDetection` are result types of the corresponding models, including classes/text, confidence, and bounding boxes/points

For a UI or API, return the following information separately:

- File name and absolute or application-relative path.
- Human-readable capture date, or an empty string when unavailable.
- Detected object names and confidence values.
- Recognized OCR text and confidence values.
- OCR quadrilateral coordinates.
- Object bounding boxes.
- Searchable combined text.

Detection boxes use their implementation-specific coordinate conventions. Do not expose or transform coordinates until the target UI coordinate system and image resize behavior are defined.

#### Database integration

Use `ingest/storage.rs::SqliteStore` when SQLite is the desired integration backend.

The current table stores detection arrays as serialized JSON. The table is suitable for a small experimental application or embedded gallery, but it is not a normalized schema for large-scale relational querying. A production integration should decide whether object detections and OCR regions should become separate tables or remain JSON fields.

The schema migration behavior should be considered when deploying the application:

- Legacy `yolo_detections_json` columns are migrated to `obj_detections_json`.
- Legacy database rows receive an empty date value unless migrated records already contain date information.
- Runtime benchmark columns are not included in `ImageRecord`.
- `file_name` is the SQLite primary key and is replaced by `INSERT OR REPLACE` for the same file name.

Applications should not assume that duplicate names from different directories are distinct. If the source directory can contain the same file name in multiple folders, migrate the key or storage contract to use a stable path or content identifier instead of `file_name` alone.

#### JSON integration

`save_records_json()` serializes `Vec<ImageRecord>` as a JSON array. JSON is most useful for export, backup, and handoff to other applications. It is not required for normal querying in the current code.

Before consuming JSON outside Rust, document or validate:

- The meaning of bounding-box coordinates.
- The coordinate basis used after image resizing.
- Whether OCR confidence is averaged or region-specific.
- Whether an empty date means missing metadata or metadata unavailable at extraction time.
- Whether empty detections mean no results or inference failure.

The existing benchmark JSON path is test-oriented:

```text
data/test/dataview.json
```

A production integration should separate benchmark output from application record export.

#### ONNX environment integration

`det_test` uses both Ultralytics object detection and OAR-OCR. It deliberately shares a process-wide ONNX Runtime environment among them.

Initialize shared ONNX state in this order:

1. Call `ort::init().with_telemetry(false).build()`.
2. Pass a clone to `oar_ocr::core::inference::set_ort_environment()`.
3. Pass the original or another clone to `ultralytics_inference::set_ort_environment()`.
4. Construct `ObjDetector` and `OcrPipeline`.

Do not create additional ONNX environments after model construction. A process should call the shared environment setup once. Tests that exercise both engines in the same process should also avoid initializing separate environments or running the setup concurrently.

The Ultralytics implementation is patchable through `det_test/patches/ultralytics-inference/`. If another Ultralytics package or direct ONNX Runtime implementation is introduced, first verify that only one ONNX Runtime environment is created in the process and that runtime versions are compatible.

#### Image and date integration

The ingestion path accepts files from a fixed image-directory argument and filters supported filename extensions before format-based decoding. `load_image()` can detect an image format independently of an extension, but it cannot recover truly corrupt or unsupported files.

The date is extracted from EXIF fields before record construction. It is not derived from the file name or filesystem modification time. An application that requires date completeness should add and document a fallback, such as:

- Preserve an empty date and distinguish missing metadata from an unknown date.
- Fall back to the filesystem creation or modification time.
- Exclude undated images from date-filtered views.

The host application should define which policy it uses. The current library behavior preserves an empty string for an unavailable date.

#### Benchmark integration

Benchmark functionality should remain separate from the production data-ingestion path.

The benchmark module currently provides:

- Per-image preprocessing, object detection, OCR, post-processing, and total runtimes.
- RSS memory and process CPU samples.
- Per-record field storage sizes.
- Database insertion timing and memory measurements.
- Search-query timing and memory measurements.
- Search hit count.
- Summary min, max, median, mean, and standard-deviation statistics.
- CSV export in `data/benchmark_profiles.csv`.

Benchmark output is intended for analysis and reproducibility. It should not be used as the authoritative runtime database or application analytics database.

When benchmark results feed another system:

1. Export machine-readable values rather than parsing terminal text.
2. Record model names, precision, thread count, image count, image resolution, CPU, operating system, and ONNX provider metadata alongside measurements.
3. Use a warm-up phase and repeat measurements.
4. Preserve the original model and dataset versions.
5. Keep benchmark database and JSON destinations separate from application data.

### Integrating `siglip2_test`

`siglip2_test` is experimental and is not required for the production `det_test` gallery workflow. It should be integrated separately if the larger application needs SigLIP-style text or vision embeddings.

The standalone runner provides:

- `TextEngine::infer(prompt)` for normalized text embeddings.
- `VisionEngine::infer(tensor)` for normalized image embeddings.
- `ImageDataset` for flat JPEG, JPG, and PNG directory iteration.
- `TelemetryGuard` for RSS and CPU timeline samples.

Before integration:

1. Verify that the vision and text models produce the expected paired embedding space.
2. Match tokenizer behavior exactly to the text model export.
3. Verify input names and output names against the ONNX models.
4. Preserve the `224 x 224`, CHW, and `pixel / 127.5 - 1.0` preprocessing contract unless both model exports are changed.
5. Keep text and vision model lifetimes explicit so models can be unloaded independently.
6. Use the existing `memory_trace.csv` only as benchmark evidence, not as application telemetry.

The SigLIP-style engines emit L2-normalized vectors. Similarity can be computed with a dot product after both vectors use the same embedding model and compatible preprocessing contracts:

```rust
let similarity: f32 = text_embedding
    .iter()
    .zip(image_embedding.iter())
    .map(|(left, right)| left * right)
    .sum();
```

Do not combine `det_test` and `siglip2_test` into one process solely by changing Cargo dependencies. Verify ONNX Runtime compatibility, session thread counts, memory limits, and environment initialization before sharing runtimes. If the goal is to integrate image embeddings into a gallery, define a separate embedding record, vector store, or index that references the existing `ImageRecord.file_path` or a stable content identifier.
