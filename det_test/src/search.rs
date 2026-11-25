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
