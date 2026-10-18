use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use sysinfo::System;

pub struct TelemetryGuard {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<Vec<(Duration, u64, f32)>>>,
    start_time: Instant,
}

impl TelemetryGuard {
    pub fn new(interval_ms: u64) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();
        let start_time = Instant::now();

        let handle = thread::spawn(move || {
            let mut sys = System::new_all();
            let pid = match sysinfo::get_current_pid() {
                Ok(pid) => pid,
                Err(_) => return Vec::new(),
            };

            let mut samples = Vec::new();

            while r.load(Ordering::Relaxed) {
                sys.refresh_process(pid);
                if let Some(process) = sys.process(pid) {
                    let bytes = process.memory();
                    let cpu_pct = process.cpu_usage();
                    let elapsed = start_time.elapsed();
                    samples.push((elapsed, bytes, cpu_pct));
                }
                thread::sleep(Duration::from_millis(interval_ms));
            }

            samples
        });

        Self {
            running,
            handle: Some(handle),
            start_time,
        }
    }

    pub fn stop(mut self) -> (u64, f32, Vec<(Duration, u64, f32)>) {
        self.running.store(false, Ordering::Relaxed);
        let samples = if let Some(handle) = self.handle.take() {
            handle.join().unwrap_or_default()
        } else {
            Vec::new()
        };

        let peak_bytes = samples.iter().map(|&(_, bytes, _)| bytes).max().unwrap_or(0);
        let peak_mb = peak_bytes / (1024 * 1024);
        let peak_cpu = samples.iter().map(|&(_, _, cpu)| cpu).fold(0.0f32, |a, b| a.max(b));
        (peak_mb, peak_cpu, samples)
    }

    /// Exports the memory and CPU timeline trace to a CSV file.
    pub fn save_csv<P: AsRef<Path>>(
        samples: &[(Duration, u64, f32)],
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "elapsed_ms,elapsed_secs,rss_bytes,rss_mb,cpu_usage_pct")?;
        for &(elapsed, bytes, cpu) in samples {
            let elapsed_ms = elapsed.as_millis();
            let elapsed_secs = elapsed.as_secs_f64();
            let rss_mb = bytes as f64 / (1024.0 * 1024.0);
            writeln!(
                writer,
                "{},{:.4},{},{:.2},{:.2}",
                elapsed_ms, elapsed_secs, bytes, rss_mb, cpu
            )?;
        }

        writer.flush()?;
        Ok(())
    }
}