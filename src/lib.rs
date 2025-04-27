pub mod angle_net;
pub mod crnn_net;
pub mod db_net;
pub mod ocr_error;
pub mod ocr_lite;
pub mod ocr_result;
pub mod ocr_utils;
pub mod scale_param;

#[cfg(test)]
mod tests {
    use crate::{ocr_error::OcrError, ocr_lite::OcrLite};

    #[test]
    fn run_test() -> Result<(), OcrError> {
        let mut ocr = OcrLite::new();
        ocr.init_models(
            "./models/ch_PP-OCRv4_det_infer.onnx",
            "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "./models/ch_PP-OCRv4_rec_infer.onnx",
            "./models/ppocr_keys_v1.txt",
            2,
        )?;

        println!("===test_1===");
        let res = ocr.detect_from_path(
            "./docs/test_images/test_1.png",
            50,
            1024,
            0.5,
            0.3,
            1.6,
            true,
            false,
        )?;
        res.text_blocks.iter().for_each(|item| {
            println!("text: {} score: {}", item.text, item.text_score);
        });
        println!("===test_2===");
        let res = ocr.detect_from_path(
            "./docs/test_images/test_2.png",
            50,
            1024,
            0.5,
            0.3,
            1.6,
            true,
            false,
        )?;
        res.text_blocks.iter().for_each(|item| {
            println!("text: {} score: {}", item.text, item.text_score);
        });

        // 通过 image 读取图片
        println!("===test_3===");
        let test_three_img = image::open("./docs/test_images/test_3.png")
            .unwrap()
            .to_rgb8();
        let res = ocr.detect(&test_three_img, 50, 1024, 0.5, 0.3, 1.6, true, false)?;
        res.text_blocks.iter().for_each(|item| {
            println!("text: {} score: {}", item.text, item.text_score);
        });

        Ok(())
    }
}
