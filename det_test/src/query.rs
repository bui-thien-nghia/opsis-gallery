pub mod search;

use crate::ingest::storage::SqliteStore;
use anyhow::{Context, Result};
use search::FuzzySearcher;
use std::io::{self, BufRead, Write};
use std::path::Path;

pub fn run_query_loop(db_path: &Path, limit: usize) -> Result<()> {
    let store = SqliteStore::open(db_path)?;
    let loaded_records = store.load_all_records()?;
    let mut searcher = FuzzySearcher::new();

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut consecutive_blanks = 0u8;

    println!("Entering query mode (enter a blank line twice to exit):");

    loop {
        print!("query> ");
        stdout.flush().context("Failed to flush stdout")?;

        let mut line = String::new();
        let bytes_read = stdin
            .lock()
            .read_line(&mut line)
            .context("Failed to read query from stdin")?;
        if bytes_read == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            consecutive_blanks += 1;
            if consecutive_blanks >= 2 {
                println!("Exiting query mode.");
                break;
            }
            continue;
        }

        consecutive_blanks = 0;
        let hits = searcher.search(&loaded_records, trimmed, limit);
        println!("Found {} hit(s) for \"{}\":", hits.len(), trimmed);
        for hit in hits.iter().take(limit) {
            println!("  [score={}] {}", hit.score, hit.record.file_path);
        }
    }

    Ok(())
}
