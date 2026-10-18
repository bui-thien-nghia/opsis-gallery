use rtbench::enc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array4;
use std::path::Path;
use enc::text::TextEngine;
use enc::vision::VisionEngine;

fn bench_vision_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vision_Preprocessing");
    let sample_image_small = Path::new("assets/images/IMG_20230825_083040.jpg"); // Small: <= 2MB
    let sample_image_medium = Path::new("assets/images/IMG_20250711_125425.jpg"); // Medium: <= 5MB
    let sample_image_large = Path::new("assets/images/IMG_20250630_100444.jpg"); // Large: <= 8MB
    if sample_image_small.exists() {
        group.bench_function("resize_and_normalize_small", |b| {
            b.iter(|| {
                let _ = VisionEngine::preprocess_image(sample_image_small);
            });
        });
    }
    if sample_image_medium.exists() {
        group.bench_function("resize_and_normalize_medium", |b| {
            b.iter(|| {
                let _ = VisionEngine::preprocess_image(sample_image_medium);
            });
        });
    }
    if sample_image_large.exists() {
        group.bench_function("resize_and_normalize_large", |b| {
            b.iter(|| {
                let _ = VisionEngine::preprocess_image(sample_image_large);
            });
        });
    }
    group.finish();
}

fn bench_vision_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vision_Encoder_Inference");
    let dummy_tensor = Array4::<f32>::zeros((1, 3, 224, 224));

    for threads in [1, 2, 4] {
        let model_path = Path::new("assets/enc/vision.onnx");
        if !model_path.exists() {
            eprintln!("Skipping vision bench: {:?} not found.", model_path);
            return;
        }

        let mut engine = VisionEngine::new(model_path, threads)
            .expect("Failed to initialize VisionEngine");

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("Threads", threads),
            &threads,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(engine.infer(black_box(dummy_tensor.clone())));
                });
            },
        );
    }
    group.finish();
}

fn bench_text_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("Text_Encoder_Inference");

    let model_path = Path::new("assets/enc/text.onnx");
    let tokenizer_path = Path::new("assets/enc/tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!("Skipping text bench: Model or tokenizer missing in assets/enc/");
        return;
    }

    let prompt = "A photo of a dog playing in the park.";

    for threads in [1, 2, 4] {
        let mut engine = TextEngine::new(model_path, tokenizer_path, threads)
            .expect("Failed to initialize TextEngine");

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("Threads", threads),
            &threads,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(engine.infer(black_box(&prompt)));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_vision_preprocessing, bench_vision_inference, bench_text_inference);
criterion_main!(benches);