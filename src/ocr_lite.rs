use crate::{
    angle_net::AngleNet,
    crnn_net::CrnnNet,
    db_net::DbNet,
    ocr_error::OcrError,
    ocr_result::{OcrResult, Point, TextBlock},
    ocr_utils::OcrUtils,
    scale_param::ScaleParam,
};

#[derive(Debug)]
pub struct OcrLite {
    db_net: DbNet,
    angle_net: AngleNet,
    crnn_net: CrnnNet,
}

impl OcrLite {
    pub fn new() -> Self {
        Self {
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
    ) -> Result<(), OcrError> {
        self.db_net.init_model(det_path, num_thread)?;
        self.angle_net.init_model(cls_path, num_thread)?;
        self.crnn_net.init_model(rec_path, keys_path, num_thread)?;
        Ok(())
    }

    pub fn detect(
        &self,
        img_src: &image::RgbImage,
        padding: u32,
        max_side_len: u32,
        box_score_thresh: f32,
        box_thresh: f32,
        un_clip_ratio: f32,
        do_angle: bool,
        most_angle: bool,
    ) -> Result<OcrResult, OcrError> {
        let origin_max_side = img_src.width().max(img_src.height());
        let mut resize;
        if max_side_len <= 0 || max_side_len > origin_max_side {
            resize = origin_max_side;
        } else {
            resize = max_side_len;
        }
        resize += 2 * padding;

        let mut padding_src = OcrUtils::make_padding(img_src, padding)?;

        let scale = ScaleParam::get_scale_param(&padding_src, resize);

        self.detect_once(
            &mut padding_src,
            &scale,
            padding,
            box_score_thresh,
            box_thresh,
            un_clip_ratio,
            do_angle,
            most_angle,
        )
    }

    pub fn detect_from_path(
        &self,
        img_path: &str,
        padding: u32,
        max_side_len: u32,
        box_score_thresh: f32,
        box_thresh: f32,
        un_clip_ratio: f32,
        do_angle: bool,
        most_angle: bool,
    ) -> Result<OcrResult, OcrError> {
        let mut img_src = image::open(img_path)?.to_rgb8();

        self.detect(
            &mut img_src,
            padding as u32,
            max_side_len as u32,
            box_score_thresh,
            box_thresh,
            un_clip_ratio,
            do_angle,
            most_angle,
        )
    }

    fn detect_once(
        &self,
        img_src: &mut image::RgbImage,
        scale: &ScaleParam,
        padding: u32,
        box_score_thresh: f32,
        box_thresh: f32,
        un_clip_ratio: f32,
        do_angle: bool,
        most_angle: bool,
    ) -> Result<OcrResult, OcrError> {
        let text_boxes = self.db_net.get_text_boxes(
            img_src,
            scale,
            box_score_thresh,
            box_thresh,
            un_clip_ratio,
        )?;

        let mut part_images = OcrUtils::get_part_images(img_src, &text_boxes);

        let angles = self
            .angle_net
            .get_angles(&part_images, do_angle, most_angle)?;

        let mut rotated_images: Vec<image::RgbImage> = Vec::with_capacity(part_images.len());
        // 用 pop 转移所有权，反转以输出正确结果
        part_images.reverse();
        for i in (0..angles.len()).rev() {
            if let Some(mut img) = part_images.pop() {
                if angles[i].index == 1 {
                    OcrUtils::mat_rotate_clock_wise_180(&mut img);
                }
                rotated_images.push(img);
            }
        }

        let text_lines = self.crnn_net.get_text_lines(&rotated_images)?;

        let mut text_blocks = Vec::with_capacity(text_lines.len());
        for i in 0..text_lines.len() {
            text_blocks.push(TextBlock {
                box_points: text_boxes[i]
                    .points
                    .iter()
                    .map(|p| Point {
                        x: ((p.x as f32) - padding as f32) as u32,
                        y: ((p.y as f32) - padding as f32) as u32,
                    })
                    .collect(),
                box_score: text_boxes[i].score,
                angle_index: angles[i].index,
                angle_score: angles[i].score,
                text: text_lines[i].text.clone(),
                text_score: text_lines[i].text_score,
            });
        }

        Ok(OcrResult { text_blocks })
    }
}
