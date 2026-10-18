mod enc;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use enc::dataset::ImageDataset;
use enc::telemetry::TelemetryGuard;
use enc::text::TextEngine;
use enc::vision::VisionEngine;
use sysinfo::{get_current_pid, System};

// Utils
fn current_rss_mb() -> u64 {
    let Ok(pid) = get_current_pid() else {
        return 0;
    };
    let mut system = System::new();
    system.refresh_process(pid);
    system
        .process(pid)
        .map(|process| process.memory() / (1024 * 1024))
        .unwrap_or(0)
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Memory usage benchmarking for SigLIP2 and CLIP-like models")]
struct Args {
    #[arg(long, default_value = "assets/enc/vision.onnx")]
    vision_model: PathBuf,

    #[arg(long, default_value = "assets/enc/text.onnx")]
    text_model: PathBuf,

    #[arg(long, default_value = "assets/enc/tokenizer.json")]
    tokenizer: PathBuf,

    #[arg(long, default_value = "assets/images")]
    image_dir: PathBuf,

    #[arg(long, default_value = "A grill party on a bench at sunset.")]
    prompt: String,

    #[arg(long, default_value_t = 2)]
    threads: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("==========================================");
    println!("           CPU BENCHMARK RUNNER           ");
    println!("==========================================");
    println!("Threads allocated: {}", args.threads);
    println!("Image directory:   {:?}", args.image_dir);
    println!();

    let mut text_rss_post_creation = 0;
    let mut text_rss_post_inf = 0;
    let mut vision_rss_post_creation = 0;
    let mut vision_rss_post_inf = 0;
    let mut rss_pre_shutdown = 0;

    let telemetry = TelemetryGuard::new(50);
    if args.text_model.exists() && args.tokenizer.exists() {
        println!("[1/2] Initializing Text Engine...");
        let mut text_engine = TextEngine::new(&args.text_model, &args.tokenizer, args.threads)?;
        text_rss_post_creation = current_rss_mb();  
        
        let prompt = args.prompt.clone();
        let prompt_repeat_count = 1000;
        println!("Running text inference for prompt {} times: \"{}\"", prompt_repeat_count, args.prompt);

        let start = Instant::now();
        for _ in 0..prompt_repeat_count {let _ = text_engine.infer(&prompt)?;}
        let duration = start.elapsed();
        text_rss_post_inf = current_rss_mb();

        println!("Text inference completed in {:.2?}", duration);
    } else {
        println!("[1/2] Skipping Text Engine (Model or tokenizer not found in assets/).");
    }

    println!();

    // Vision model bench
    if args.vision_model.exists() {
        println!("[2/2] Initializing Vision Engine...");
        let mut vision_engine = VisionEngine::new(&args.vision_model, args.threads)?;
        vision_rss_post_creation = current_rss_mb();

        let dataset = ImageDataset::new(&args.image_dir)?;
        let total_images = dataset.len();

        if total_images == 0 {
            println!("No valid images (.jpg, .jpeg, .png) found in {:?}.", args.image_dir);
        } else {
            println!("Processing {} images...", total_images);
            
            let start_total = Instant::now();
            let mut total_preprocess_time = std::time::Duration::ZERO;
            let mut total_inference_time = std::time::Duration::ZERO;
            let mut processed_count = 0;

            for image_path in dataset {
                // Preprocessing timer
                let prep_start = Instant::now();
                let tensor_res = VisionEngine::preprocess_image(&image_path);
                let prep_duration = prep_start.elapsed();
                match tensor_res {
                    Ok(tensor) => {
                        // Inference + Postprocessing timer
                        let inf_start = Instant::now();
                        let inf_res = vision_engine.infer(tensor);
                        let inf_duration = inf_start.elapsed();
                        match inf_res {
                            Ok(_) => {
                                total_preprocess_time += prep_duration;
                                total_inference_time += inf_duration;
                                processed_count += 1;
                                if processed_count % 10 == 0 || processed_count == total_images {
                                    print!("\rProgress: {}/{} images processed", processed_count, total_images);
                                    std::io::Write::flush(&mut std::io::stdout())?;
                                }
                            }
                            Err(e) => eprintln!("\nInference failed for {:?}: {}", image_path, e),
                        }
                    }
                    Err(e) => eprintln!("\nPreprocessing failed for {:?}: {}", image_path, e),
                }
            }
            vision_rss_post_inf = current_rss_mb();

            let total_duration = start_total.elapsed();

            println!("\n\n--- Vision Timing Breakdown ---");
            println!("Total time:    {:.2?}", total_duration);
            println!("Processed Images:       {}", processed_count);
            if processed_count > 0 {
                println!("Total Preprocessing:    {:.2?}", total_preprocess_time);
                println!("  Avg per image:        {:.2?}", total_preprocess_time / processed_count as u32);
                println!("Total ONNX Inference:   {:.2?}", total_inference_time);
                println!("  Avg per image:        {:.2?}", total_inference_time / processed_count as u32);
                
                let grand_total = total_preprocess_time + total_inference_time;
                let overall_fps = processed_count as f64 / grand_total.as_secs_f64();
                println!("Pipeline Throughput:    {:.2} FPS", overall_fps);
            }
        }
    } else {
        println!("[2/2] Skipping Vision Engine (Model not found).");
    }

    rss_pre_shutdown = current_rss_mb();
    let (peak_memory_mb, peak_cpu_usage,rss_seq) = telemetry.stop();
    let csv_path = "memory_trace.csv";

    println!();
    println!("==========================================");
    println!("       BENCHMARK SUMMARY REPORT           ");
    println!("==========================================");
    println!("Peak CPU Utilization:  {:.1}%", peak_cpu_usage);
    println!("Peak RSS Memory Usage    : {} MB", peak_memory_mb);
    println!("Text RSS Post-load       : {} MB", text_rss_post_creation);
    println!("Text RSS Post-inference  : {} MB", text_rss_post_inf);
    println!("Vision RSS Post-load     : {} MB", vision_rss_post_creation);
    println!("Vision RSS Post-inference: {} MB", vision_rss_post_inf);
    println!("RSS Pre-shutdown         : {} MB", rss_pre_shutdown);
    if let Err(e) = TelemetryGuard::save_csv(&rss_seq, csv_path) {
        eprintln!("Failed to save CSV trace: {}", e);
    } else {
        println!("Saved memory timeline trace ({} samples) to '{}'", rss_seq.len(), csv_path);
    }
    println!("==========================================");

    Ok(())
}