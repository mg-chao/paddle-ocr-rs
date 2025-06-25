pub mod angle_net;
pub mod base_net;
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
    use std::fs;
    use std::io::{Cursor, Read};

    #[test]
    fn run_test() -> Result<(), OcrError> {
        let mut ocr = OcrLite::new();
        ocr.init_models(
            "./models/ch_PP-OCRv5_mobile_det.onnx",
            "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "./models/ch_PP-OCRv5_rec_mobile_infer.onnx",
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
            false,
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
            false,
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

    #[test]
    fn run_test_from_memory() -> Result<(), OcrError> {
        let det_bytes = fs::read("./models/ch_PP-OCRv4_det_infer.onnx")?;
        let cls_bytes = fs::read("./models/ch_ppocr_mobile_v2.0_cls_infer.onnx")?;
        let rec_bytes = fs::read("./models/ch_PP-OCRv4_rec_infer.onnx")?;

        let mut ocr = OcrLite::new();
        ocr.init_models_from_memory(&det_bytes, &cls_bytes, &rec_bytes, 2)?;

        println!("===test_from_memory===");
        let test_img = image::open("./docs/test_images/test_1.png")
            .unwrap()
            .to_rgb8();
        let res = ocr.detect(&test_img, 50, 1024, 0.5, 0.3, 1.6, false, false)?;
        res.text_blocks.iter().for_each(|item| {
            println!("text: {} score: {}", item.text, item.text_score);
        });

        Ok(())
    }

    #[test]
    fn run_test_from_cursor() -> Result<(), OcrError> {
        let mut det_file = fs::File::open("./models/ch_PP-OCRv4_det_infer.onnx")?;
        let mut cls_file = fs::File::open("./models/ch_ppocr_mobile_v2.0_cls_infer.onnx")?;
        let mut rec_file = fs::File::open("./models/ch_PP-OCRv4_rec_infer.onnx")?;

        let mut det_buffer = Vec::new();
        let mut cls_buffer = Vec::new();
        let mut rec_buffer = Vec::new();

        det_file.read_to_end(&mut det_buffer)?;
        cls_file.read_to_end(&mut cls_buffer)?;
        rec_file.read_to_end(&mut rec_buffer)?;

        let det_cursor = Cursor::new(det_buffer);
        let cls_cursor = Cursor::new(cls_buffer);
        let rec_cursor = Cursor::new(rec_buffer);

        let det_bytes = det_cursor.into_inner();
        let cls_bytes = cls_cursor.into_inner();
        let rec_bytes = rec_cursor.into_inner();

        let mut ocr = OcrLite::new();
        ocr.init_models_from_memory(&det_bytes, &cls_bytes, &rec_bytes, 2)?;

        println!("===test_from_cursor===");
        let test_img = image::open("./docs/test_images/test_2.png")
            .unwrap()
            .to_rgb8();
        let res = ocr.detect(&test_img, 50, 1024, 0.5, 0.3, 1.6, false, false)?;
        res.text_blocks.iter().for_each(|item| {
            println!("text: {} score: {}", item.text, item.text_score);
        });

        Ok(())
    }

    #[test]
    fn run_test_angle_rollback() -> Result<(), OcrError> {
        let mut ocr = OcrLite::new();
        ocr.init_models(
            "./models/ch_PP-OCRv4_det_infer.onnx",
            "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "./models/ch_PP-OCRv4_rec_infer.onnx",
            2,
        )?;

        println!("===test_angle_ori===");
        let test_img = image::open("./docs/test_images/test_4.png")
            .unwrap()
            .to_rgb8();
        let res = ocr.detect(&test_img, 50, 1024, 0.5, 0.3, 1.6, true, false)?;
        res.text_blocks.iter().for_each(|item| {
            println!("text: {} score: {}", item.text, item.text_score);
        });

        println!("===test_angle_rollback===");
        let test_img = image::open("./docs/test_images/test_4.png")
            .unwrap()
            .to_rgb8();
        let res =
            ocr.detect_angle_rollback(&test_img, 50, 1024, 0.5, 0.3, 1.6, true, false, 0.8)?;
        res.text_blocks.iter().for_each(|item| {
            println!("text: {} score: {}", item.text, item.text_score);
        });

        Ok(())
    }

    #[test]
    fn run_test_from_custom() -> Result<(), OcrError> {
        let mut ocr = OcrLite::new();
        ocr.init_models_custom(
            "./models/ch_PP-OCRv5_mobile_det.onnx",
            "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "./models/ch_PP-OCRv5_rec_mobile_infer.onnx",
            |builder| Ok(builder.with_inter_threads(2)?.with_intra_threads(2)?),
        )?;

        println!("===test_from_custom===");
        let res = ocr.detect_from_path(
            "./docs/test_images/test_4.png",
            50,
            1024,
            0.5,
            0.3,
            1.6,
            false,
            false,
        )?;
        res.text_blocks.iter().for_each(|item| {
            println!("text: {} score: {}", item.text, item.text_score);
        });

        Ok(())
    }
}
