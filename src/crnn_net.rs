use ort::inputs;
use ort::session::Session;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::{base_net::BaseNet, ocr_error::OcrError, ocr_result::TextLine, ocr_utils::OcrUtils};

const CRNN_DST_HEIGHT: u32 = 48;
const MEAN_VALUES: [f32; 3] = [127.5, 127.5, 127.5];
const NORM_VALUES: [f32; 3] = [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5];

#[derive(Debug)]
pub struct CrnnNet {
    session: Option<Session>,
    keys: Vec<String>,
    input_names: Vec<String>,
}

impl BaseNet for CrnnNet {
    fn new() -> Self {
        Self {
            session: None,
            keys: Vec::new(),
            input_names: Vec::new(),
        }
    }

    fn set_input_names(&mut self, input_names: Vec<String>) {
        self.input_names = input_names;
    }

    fn set_session(&mut self, session: Option<Session>) {
        self.session = session;
    }
}

impl CrnnNet {
    pub fn init_model(
        &mut self,
        path: &str,
        keys_path: &str,
        num_thread: usize,
    ) -> Result<(), OcrError> {
        BaseNet::init_model(self, path, num_thread)?;

        self.keys = self.get_keys(keys_path)?;

        Ok(())
    }

    pub fn init_model_from_memory(
        &mut self,
        model_bytes: &[u8],
        keys_bytes: &[u8],
        num_thread: usize,
    ) -> Result<(), OcrError> {
        BaseNet::init_model_from_memory(self, model_bytes, num_thread)?;

        self.keys = self.get_keys_from_memory(keys_bytes)?;

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

    fn get_keys_from_memory(&mut self, keys_bytes: &[u8]) -> Result<Vec<String>, OcrError> {
        let mut keys = Vec::new();

        keys.push("#".to_string());

        let reader = BufReader::new(keys_bytes);

        for line in reader.lines() {
            keys.push(line?);
        }

        keys.push(" ".to_string());

        Ok(keys)
    }

    pub fn get_text_lines(
        &self,
        part_imgs: &Vec<image::RgbImage>,
        angle_rollback_records: &HashMap<usize, image::RgbImage>,
        angle_rollback_threshold: f32,
    ) -> Result<Vec<TextLine>, OcrError> {
        let mut text_lines = Vec::new();

        for (index, img) in part_imgs.iter().enumerate() {
            let mut text_line = self.get_text_line(img)?;

            if text_line.text_score.is_nan() || text_line.text_score < angle_rollback_threshold {
                if let Some(angle_rollback_record) = angle_rollback_records.get(&index) {
                    text_line = self.get_text_line(angle_rollback_record)?;
                }
            }

            text_lines.push(text_line);
        }

        Ok(text_lines)
    }

    fn get_text_line(&self, img_src: &image::RgbImage) -> Result<TextLine, OcrError> {
        let Some(session) = &self.session else {
            return Err(OcrError::SessionNotInitialized);
        };

        let scale = CRNN_DST_HEIGHT as f32 / img_src.height() as f32;
        let dst_width = (img_src.width() as f32 * scale) as u32;

        let src_resize = image::imageops::resize(
            img_src,
            dst_width as u32,
            CRNN_DST_HEIGHT as u32,
            image::imageops::FilterType::Nearest,
        );

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
