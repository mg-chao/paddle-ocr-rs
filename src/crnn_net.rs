use opencv as cv;
use opencv::core::{Mat, Size};
use opencv::prelude::*;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::ocr_error::OcrError;
use crate::ocr_result::TextLine;
use crate::ocr_utils::OcrUtils;

const CRNN_DST_HEIGHT: i32 = 48;
const MEAN_VALUES: [f32; 3] = [127.5, 127.5, 127.5];
const NORM_VALUES: [f32; 3] = [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5];

#[derive(Debug)]
pub struct CrnnNet {
    session: Option<Session>,
    keys: Vec<String>,
    input_names: Vec<String>,
}

impl CrnnNet {
    pub fn new() -> Self {
        Self {
            session: None,
            keys: Vec::new(),
            input_names: Vec::new(),
        }
    }

    pub fn init_model(
        &mut self,
        path: &str,
        keys_path: &str,
        num_thread: usize,
    ) -> Result<(), OcrError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level2)?
            .with_intra_threads(num_thread)?
            .with_inter_threads(num_thread)?
            .commit_from_file(path)?;

        let input_names: Vec<String> = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        self.input_names = input_names;
        self.session = Some(session);

        self.keys = self.get_keys(keys_path)?;

        Ok(())
    }

    fn get_keys<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<String>, OcrError> {
        let mut keys = Vec::new();

        keys.push("#".to_string());

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            keys.push(line?);
        }

        keys.push(" ".to_string());

        Ok(keys)
    }

    pub fn get_text_lines(&self, part_imgs: &Vec<Mat>) -> Result<Vec<TextLine>, OcrError> {
        let mut text_lines = Vec::new();

        for img in part_imgs {
            let text_line = self.get_text_line(img)?;
            text_lines.push(text_line);
        }

        Ok(text_lines)
    }

    fn get_text_line(&self, src: &Mat) -> Result<TextLine, OcrError> {
        let Some(session) = &self.session else {
            return Err(OcrError::SessionNotInitialized);
        };

        let scale = CRNN_DST_HEIGHT as f32 / src.rows() as f32;
        let dst_width = (src.cols() as f32 * scale) as i32;

        let mut src_resize = Mat::default();
        cv::imgproc::resize(
            &src,
            &mut src_resize,
            Size::new(dst_width, CRNN_DST_HEIGHT),
            0.0,
            0.0,
            cv::imgproc::INTER_LINEAR,
        )
        .unwrap();

        let input_tensors =
            OcrUtils::substract_mean_normalize(&src_resize, &MEAN_VALUES, &NORM_VALUES);

        let outputs = session.run(inputs![self.input_names[0].clone() => input_tensors]?)?;

        let (_, red_data) = outputs.iter().next().unwrap();

        let src_data = red_data.try_extract_tensor::<f32>()?;

        let dimensions = src_data.shape();
        let height = dimensions[1];
        let width = dimensions[2];
        let src_data: Vec<f32> = src_data.iter().map(|&x| x).collect();

        self.score_to_text_line(&src_data, height, width)
    }

    fn score_to_text_line(
        &self,
        output_data: &Vec<f32>,
        height: usize,
        width: usize,
    ) -> Result<TextLine, OcrError> {
        let mut text_line = TextLine::default();
        let mut last_index = 0;

        let mut text_score_sum = 0.0;
        let mut text_socre_count = 0;
        for i in 0..height {
            let start = i * width;
            let stop = (i + 1) * width;
            let slice = &output_data[start..stop.min(output_data.len())];

            let (max_index, max_value) =
                slice
                    .iter()
                    .enumerate()
                    .fold((0, f32::MIN), |(max_idx, max_val), (idx, &val)| {
                        if val > max_val {
                            (idx, val)
                        } else {
                            (max_idx, max_val)
                        }
                    });

            if max_index > 0 && max_index < self.keys.len() && !(i > 0 && max_index == last_index) {
                text_line.text.push_str(&self.keys[max_index]);
                text_score_sum += max_value;
                text_socre_count += 1;
            }
            last_index = max_index;
        }

        text_line.text_score = text_score_sum / text_socre_count as f32;
        Ok(text_line)
    }
}
