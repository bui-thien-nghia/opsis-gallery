use std::fs;
use std::path::{Path, PathBuf};
use std::io;

pub struct ImageDataset {
    paths: Vec<PathBuf>,
    index: usize,
}

impl ImageDataset {
    pub fn new<P: AsRef<Path>>(dir: P) -> io::Result<Self> {
        let path = dir.as_ref();
        let mut paths = Vec::new();
        
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let file_path = entry.path();
                if file_path.is_file() {
                    if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
                        let ext_lower = ext.to_lowercase();
                        if ext_lower == "jpg" || ext_lower == "jpeg" || ext_lower == "png" {
                            paths.push(file_path);
                        }
                    }
                }
            }
        }
        
        paths.sort();
        Ok(Self { paths, index: 0 })
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

impl Iterator for ImageDataset {
    type Item = PathBuf;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.paths.len() {
            let path = self.paths[self.index].clone();
            self.index += 1;
            Some(path)
        } else {
            None
        }
    }
}