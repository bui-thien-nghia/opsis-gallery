use crate::types::ImageRecord;
use nucleo_matcher::{
    Config, Matcher, Utf32Str,
    pattern::{CaseMatching, Normalization, Pattern},
};

pub struct SearchHit<'a> {
    pub record: &'a ImageRecord,
    pub score: u32,
}

pub struct FuzzySearcher {
    matcher: Matcher,
}

impl Default for FuzzySearcher {
    fn default() -> Self {
        Self::new()
    }
}

impl FuzzySearcher {
    pub fn new() -> Self {
        Self {
            matcher: Matcher::new(Config::DEFAULT),
        }
    }

    pub fn search<'a>(
        &mut self,
        records: &'a [ImageRecord],
        query: &str,
        limit: usize,
    ) -> Vec<SearchHit<'a>> {
        let trimmed = query.trim();
        if trimmed.is_empty() || limit == 0 {
            return Vec::new();
        }

        let pattern = Pattern::parse(trimmed, CaseMatching::Ignore, Normalization::Smart);
        let mut buf = Vec::new();
        let mut hits = Vec::new();

        for record in records {
            let haystack = Utf32Str::new(&record.combined_text, &mut buf);
            if let Some(score) = pattern.score(haystack, &mut self.matcher) {
                hits.push(SearchHit { record, score });
            }
        }

        hits.sort_by(|a, b| b.score.cmp(&a.score));
        hits.truncate(limit);
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StageRuntimes;

    fn record_with_text(name: &str, combined_text: &str) -> ImageRecord {
        ImageRecord {
            file_name: name.to_string(),
            file_path: name.to_string(),
            object_classes: String::new(),
            text_detections: String::new(),
            combined_text: combined_text.to_string(),
            yolo_detections: Vec::new(),
            ocr_detections: Vec::new(),
            runtime: StageRuntimes {
                preprocessing: 0.0,
                yolo_inference: 0.0,
                ocr_inference: 0.0,
                postprocessing: 0.0,
                total: 0.0,
            },
        }
    }

    #[test]
    fn search_handles_empty_query_and_zero_limit() {
        let records = vec![record_with_text("a.jpg", "person on a boat")];
        let mut searcher = FuzzySearcher::new();
        assert!(searcher.search(&records, "   ", 10).is_empty());
        assert!(searcher.search(&records, "boat", 0).is_empty());
    }

    #[test]
    fn search_matches_and_respects_limit() {
        let records = vec![
            record_with_text("a.jpg", "person on a boat"),
            record_with_text("b.jpg", "cat on a sofa"),
            record_with_text("c.jpg", "fishing boat harbor"),
        ];
        let mut searcher = FuzzySearcher::new();
        let hits = searcher.search(&records, "boat", 1);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].record.combined_text.contains("boat"));
    }
}