use std::path::Path;
use tokenizers::Tokenizer;
use ndarray::Array2;
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::{TensorRef}};

pub struct TextEngine {
    session: Session,
    tokenizer: Tokenizer,
}

impl TextEngine {
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        model_path: P1,
        tokenizer_path: P2,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_path)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self { session, tokenizer })
    }

    pub fn tokenize_prompt(&self, prompt: &str) -> Result<Array2<i64>, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let ids = encoding.get_ids();
        let seq_len = ids.len();
        let mut input_ids = Array2::<i64>::zeros((1, seq_len));
        for (j, &id) in ids.iter().enumerate() {
            input_ids[[0, j]] = id as i64;
        }

        Ok(input_ids)
    }

    pub fn infer(&mut self, prompt: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let input_ids = self.tokenize_prompt(prompt)?;
        let input_ids_ref = TensorRef::from_array_view(&input_ids)?;
        let outputs = self.session.run(ort::inputs!["input_ids" => input_ids_ref])?;

        // let array = outputs[0].try_extract_array::<f32>()?;
        let array = outputs
            .get("pooler_output")
            .ok_or("Outputs has no attribute \"pooler_output\"")?
            .try_extract_array()?;
        let mut embedding: Vec<f32> = array.iter().cloned().collect();
        crate::enc::vision::l2_normalize(&mut embedding);

        Ok(embedding)
    }
}