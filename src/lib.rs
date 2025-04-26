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
        let res =
            ocr.detect_from_path("./test/test_1.png", 50, 1024, 0.5, 0.3, 1.6, true, false)?;
        println!("res: {}", res);
        println!("===test_2===");
        let res =
            ocr.detect_from_path("./test/test_2.png", 50, 1024, 0.5, 0.3, 1.6, true, false)?;
        println!("res: {}", res);

        Ok(())
    }
}
