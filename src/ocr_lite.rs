use opencv::{
    core::{Mat, Rect},
    imgcodecs::{imread, IMREAD_COLOR},
    prelude::*,
};
use std::time::Instant;

use crate::{
    angle_net::AngleNet,
    crnn_net::CrnnNet,
    db_net::DbNet,
    ocr_result::{OcrResult, TextBlock},
    ocr_utils::OcrUtils,
    scale_param::ScaleParam,
};

#[derive(Debug)]
pub struct OcrLite {
    is_part_img: bool,
    is_debug_img: bool,
    db_net: DbNet,
    angle_net: AngleNet,
    crnn_net: CrnnNet,
}

impl OcrLite {
    pub fn new() -> Self {
        Self {
            is_part_img: false,
            is_debug_img: false,
            db_net: DbNet::new(),
            angle_net: AngleNet::new(),
            crnn_net: CrnnNet::new(),
        }
    }

    pub fn init_models(
        &mut self,
        det_path: &str,
        cls_path: &str,
        rec_path: &str,
        keys_path: &str,
        num_thread: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.db_net.init_model(det_path, num_thread)?;
        self.angle_net.init_model(cls_path, num_thread)?;
        self.crnn_net.init_model(rec_path, keys_path, num_thread)?;
        Ok(())
    }

    pub fn detect(
        &self,
        img_path: &str,
        padding: i32,
        max_side_len: i32,
        box_score_thresh: f32,
        box_thresh: f32,
        un_clip_ratio: f32,
        do_angle: bool,
        most_angle: bool,
    ) -> Result<OcrResult, Box<dyn std::error::Error>> {
        let origin_src = imread(img_path, IMREAD_COLOR)?;

        let origin_max_side = origin_src.cols().max(origin_src.rows());

        let resize = if max_side_len <= 0 || max_side_len > origin_max_side {
            origin_max_side
        } else {
            max_side_len
        } + 2 * padding;

        let padding_rect = Rect::new(padding, padding, origin_src.cols(), origin_src.rows());
        let padding_src = OcrUtils::make_padding(&origin_src, padding);

        let scale = ScaleParam::get_scale_param(&padding_src, resize);

        self.detect_once(
            &padding_src,
            padding_rect,
            &scale,
            box_score_thresh,
            box_thresh,
            un_clip_ratio,
            do_angle,
            most_angle,
        )
    }

    fn detect_once(
        &self,
        src: &Mat,
        origin_rect: Rect,
        scale: &ScaleParam,
        box_score_thresh: f32,
        box_thresh: f32,
        un_clip_ratio: f32,
        do_angle: bool,
        most_angle: bool,
    ) -> Result<OcrResult, Box<dyn std::error::Error>> {
        let mut text_box_padding_img = src.clone();
        let thickness = OcrUtils::get_thickness(src);
        println!("=====Start detect=====");
        let start_time = Instant::now();

        println!("---------- step: dbNet getTextBoxes ----------");
        let text_boxes =
            self.db_net
                .get_text_boxes(src, scale, box_score_thresh, box_thresh, un_clip_ratio)?;
        let db_net_time = start_time.elapsed().as_secs_f32() * 1000.0;

        println!("TextBoxesSize({})", text_boxes.len());
        for box_item in &text_boxes {
            println!("{:?}", box_item);
        }

        println!("---------- step: drawTextBoxes ----------");
        OcrUtils::draw_text_boxes(&mut text_box_padding_img, &text_boxes, thickness);

        // Get part images
        let part_images = OcrUtils::get_part_images(src, &text_boxes);
        if self.is_part_img {
            for (i, img) in part_images.iter().enumerate() {
                opencv::highgui::imshow(&format!("PartImg({})", i), img)?;
            }
        }

        println!("---------- step: angleNet getAngles ----------");
        let angles = self
            .angle_net
            .get_angles(&part_images, do_angle, most_angle)?;

        println!("AnglesSize({})", angles.len());
        for angle_item in &angles {
            println!("{:?}", angle_item);
        }

        // Rotate part images
        let mut rotated_images = Vec::with_capacity(part_images.len());
        for (i, img) in part_images.iter().enumerate() {
            let mut cloned_img = img.clone();
            if angles[i].index == 1 {
                OcrUtils::mat_rotate_clock_wise_180(&mut cloned_img);
            }

            if self.is_debug_img {
                opencv::highgui::imshow(&format!("DebugImg({})", i), &cloned_img)?;
            }
            rotated_images.push(cloned_img);
        }

        println!("---------- step: crnnNet getTextLines ----------");
        let text_lines = self.crnn_net.get_text_lines(&rotated_images)?;

        let mut text_blocks = Vec::with_capacity(text_lines.len());
        for i in 0..text_lines.len() {
            text_blocks.push(TextBlock {
                box_points: text_boxes[i].points.clone(),
                box_score: text_boxes[i].score,
                angle_index: angles[i].index,
                angle_score: angles[i].score,
                angle_time: angles[i].time,
                text: text_lines[i].text.clone(),
                char_scores: text_lines[i].char_scores.clone(),
                crnn_time: text_lines[i].time,
                block_time: angles[i].time + text_lines[i].time,
            });
        }

        let full_detect_time = start_time.elapsed().as_secs_f32() * 1000.0;

        // Crop to original size
        let box_img = Mat::roi(&text_box_padding_img, origin_rect)?.clone_pointee();

        let str_res = text_blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<&str>>()
            .join("\n");

        println!("{}", str_res);

        Ok(OcrResult {
            text_blocks,
            db_net_time,
            box_img,
            detect_time: full_detect_time,
            str_res,
        })
    }
}
