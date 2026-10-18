# Opsis gallery's model experiment playground

## 1. Overview

This folder contains model quantization code in Python and model CPU runtime benchmarks in Rust. **The benchmark environment configuration is far from perfect.**
All models are in the `bench/assets/` folder, the rest (including SigLIP2 and MobileCLIP2) is left uncommitted due to big file size.

## 2. Details

### 2.1. YOLOE-26

YOLOE-26 allows flexible amount of detection classes, but the trade-off in accuracy is risky for fuzzy search.
Notes from runs:

- The model configuration is set as follows:

```python
imgsz=320,
quantize=8,
simplify=True
```

- At nano size (yoloe26n), the model performs not so well both prompt-free and with custom classes: classes are mostly either gibberish or not unique across images. It gets only relatively useful from the small size onward.
- Runtime is tested on a folder of 1000 images whose size is from 2.7 - 14.8 MB. The average, median, and standard deviation of processing time, from pre- to post-processing, is ~180ms, ~160ms, and ~100ms respectively.
- Setup on both Python and Rust in 2 different approaches (using the native Ultralytics library and using ONNX Runtime execution engine) differ inconsiderably in runtime (hence the setup in Rust still using `ultralytics-inference` library).
- The benchmark suggests that the total processing time pre- and post-inference is longer than the inference time itself: inference time revolves around 30ms, while total processing time stretches up to 180ms, as report has shown.

### 2.2. PaddleOCR V4

This may be caused by wrong setup or incompatible ONNX files since PaddleOCR has too many distinct versions for various languages, but detection yields nothing across 1000 images
The detection time revolves around 63ms. The total processing time should be less than 100ms.
