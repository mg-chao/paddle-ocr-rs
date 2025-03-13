mod angle_net;
mod crnn_net;
mod db_net;
mod ocr_lite;
mod ocr_result;
mod ocr_utils;
mod scale_param;

use crate::ocr_lite::OcrLite;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ocr = OcrLite::new();
    ocr.init_models(
        "./models/ch_PP-OCRv4_det_infer.onnx",
        "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "./models/ch_PP-OCRv4_rec_infer.onnx",
        "./models/ppocr_keys_v1.txt",
        4,
    )?;
    ocr.detect("./test/test_1.png", 50, 1024, 0.5, 0.3, 1.6, true, false)?;
    ocr.detect("./test/test_2.png", 50, 1024, 0.5, 0.3, 1.6, true, false)?;

    Ok(())
}