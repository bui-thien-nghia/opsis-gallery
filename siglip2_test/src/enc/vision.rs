use std::path::Path;
use image::{imageops::FilterType};
use ndarray::Array4;
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::TensorRef};
pub struct VisionEngine {
    session: Session,
}

impl VisionEngine {
    pub fn new<P: AsRef<Path>>(model_path: P, threads: usize) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    pub fn preprocess_image(path: &Path) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        let img = image::open(path)?;
        let rgb_img = img.resize_exact(224, 224, FilterType::Triangle).to_rgb8();
        let mut data = vec![0.0_f32; 3 * 224 * 224];
        let plane_size = 224 * 224;

        for (pixel_index, pixel) in rgb_img.as_raw().chunks_exact(3).enumerate() {
            data[pixel_index] = pixel[0] as f32 / 127.5 - 1.0;
            data[plane_size + pixel_index] = pixel[1] as f32 / 127.5 - 1.0;
            data[2 * plane_size + pixel_index] = pixel[2] as f32 / 127.5 - 1.0;
        }

        Ok(Array4::from_shape_vec((1, 3, 224, 224), data)?)
    }

pub fn infer(&mut self, input_tensor: Array4<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let tensor_ref = TensorRef::from_array_view(&input_tensor)?;
        let outputs = self.session.run(ort::inputs![tensor_ref])?;
        
        let array = outputs
            .get("pooler_output")
            .ok_or("Outputs has no attribute \"pooler_output\"")?
            .try_extract_array()?;
        let mut embedding: Vec<f32> = array.iter().cloned().collect();
        l2_normalize(&mut embedding);
        Ok(embedding)
    }
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}