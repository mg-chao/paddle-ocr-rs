use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session, SessionOutputs},
};

use crate::{ocr_error::OcrError, ocr_result::Angle, ocr_utils::OcrUtils};

const MEAN_VALUES: [f32; 3] = [127.5, 127.5, 127.5];
const NORM_VALUES: [f32; 3] = [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5];
const ANGLE_DST_WIDTH: u32 = 192;
const ANGLE_DST_HEIGHT: u32 = 48;
const ANGLE_COLS: usize = 2;

#[derive(Debug)]
pub struct AngleNet {
    session: Option<Session>,
    input_names: Vec<String>,
}

impl AngleNet {
    pub fn new() -> Self {
        Self {
            session: None,
            input_names: Vec::new(),
        }
    }

    pub fn init_model(&mut self, path: &str, num_thread: usize) -> Result<(), OcrError> {
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

        Ok(())
    }

    pub fn get_angles(
        &self,
        part_imgs: &[image::RgbImage],
        do_angle: bool,
        most_angle: bool,
    ) -> Result<Vec<Angle>, OcrError> {
        let mut angles = Vec::new();

        if do_angle {
            for img in part_imgs {
                let angle = self.get_angle(img)?;
                angles.push(angle);
            }
        } else {
            angles.extend(part_imgs.iter().map(|_| Angle::default()));
        }

        if do_angle && most_angle {
            let sum: i32 = angles.iter().map(|x| x.index).sum();
            let half_percent = angles.len() as f32 / 2.0;
            let most_angle_index = if (sum as f32) < half_percent { 0 } else { 1 };

            for angle in angles.iter_mut() {
                angle.index = most_angle_index;
            }
        }

        Ok(angles)
    }

    fn get_angle(&self, img_src: &image::RgbImage) -> Result<Angle, OcrError> {
        let angle;

        let Some(session) = &self.session else {
            return Err(OcrError::SessionNotInitialized);
        };

        let angle_img = image::imageops::resize(
            img_src,
            ANGLE_DST_WIDTH as u32,
            ANGLE_DST_HEIGHT as u32,
            image::imageops::FilterType::Nearest,
        );

        let input_tensors =
            OcrUtils::substract_mean_normalize(&angle_img, &MEAN_VALUES, &NORM_VALUES);

        let outputs = session.run(inputs![self.input_names[0].clone() => input_tensors]?)?;

        angle = self.score_to_angle(&outputs, ANGLE_COLS)?;

        Ok(angle)
    }

    fn score_to_angle(
        &self,
        output_tensor: &SessionOutputs,
        angle_cols: usize,
    ) -> Result<Angle, OcrError> {
        let (_, red_data) = output_tensor.iter().next().unwrap();

        let src_data: Vec<f32> = red_data
            .try_extract_tensor::<f32>()?
            .iter()
            .map(|&x| x)
            .collect();

        let mut angle = Angle::default();
        let mut max_value = f32::MIN;
        let mut angle_index = 0;

        for (i, value) in src_data.iter().take(angle_cols).enumerate() {
            if i == 0 || value > &max_value {
                max_value = *value;
                angle_index = i as i32;
            }
        }

        angle.index = angle_index;
        angle.score = max_value;
        Ok(angle)
    }
}
