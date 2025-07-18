[简体中文](./docs/README_zh-Hans.md)

## paddle-ocr-rs

Use Rust to call Paddle OCR models via ONNX Runtime for image text recognition.

### Example

```rust
use crate::{ocr_error::OcrError, ocr_lite::OcrLite};

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

// 某些情况下角度纠正会得出错误结果，支持角度纠正回退，当角度纠正后的文本识别得分低于指定值（或为 NaN）时，将使用进行角度纠正前的图片进行识别
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
```

### Reference Development Environment

| Dependency | Version                       |
| ---------- | ----------------------------- |
| rustc      | 1.84.1 (e71f9a9a9 2025-01-27) |
| cargo      | 1.84.1 (66221abde 2024-11-19) |
| OS         | Windows 11 24H2               |
| Paddle OCR | 4                             |

### Documentation

[Error Handling](/docs/error/index.md)

[Static Linking Reference](/docs/staticLinking/index.md)

### Model Source

[RapidOCR Docs](https://rapidai.github.io/RapidOCRDocs/main/model_list/)

### Related Notes

The code is referenced from [RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx), and has replaced OpenCV with image and imageproc libraries for image-related implementations.

### Results Demonstration

#### test_1.png

![test_1](/docs/test_images/test_1.png)

```bash
text: 使用Rust 通过ONNX Runtime 调用 Paddle OCR 模型进行图片文字识别。 score: 0.95269924
text: paddle-ocr-rs score: 0.9979071
```

#### test_2.png

![test_2](/docs/test_images/test_2.png)

```bash
text: 母婴用品连锁 score: 0.99713486
```

#### test_3.png

![test_3](/docs/test_images/test_3.png)

#### Output Preview

```bash
text: salta sobre o cao preguicoso. score: 0.9794339
text: perezoso. A raposa marrom rapida score: 0.9970329
text: marron rapido salta sobre el perro score: 0.9995695
text: salta sopra il cane pigro. El zorro score: 0.99923337
text: paresseux. La volpe marrone rapida score: 0.9991456
text: 《rapide> saute par-dessus le chien score: 0.9685502
text: uber den faulen Hund. Le renard brun score: 0.988613
text: Der ,schnelle" braune Fuchs springt score: 0.97560924
text: from aspammer@website.com is spam. score: 0.98167914
text: & duck/goose, as 12.5% of E-mail score: 0.98472834
text: Over the $43,456.78 <lazy> #90 dog score: 0.9847551
text: The (quick) [brown] {fox} jumps! score: 0.98300403
```
