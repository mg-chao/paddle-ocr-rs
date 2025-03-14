[简体中文](./docs/README_zh-Hans.md)

## paddle-ocr-rs

A Rust implementation for text recognition in images using Paddle OCR models through ONNX Runtime.

#### Example

```rust
use paddle_ocr_rs::{ocr_error::OcrError, ocr_lite::OcrLite, ocr_utils::OcrUtils};

fn main() -> Result<(), OcrError> {
    let mut ocr = OcrLite::new();
    ocr.init_models(
        "./models/ch_PP-OCRv4_det_infer.onnx",
        "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "./models/ch_PP-OCRv4_rec_infer.onnx",
        "./models/ppocr_keys_v1.txt",
        2,
    )?;

    // 从图片路径检测
    println!("===test_1===");
    let res = ocr.detect_from_path("./test/test_1.png", 50, 1024, 0.5, 0.3, 1.6, true, false)?;
    println!("res: {}", res);

    // 更常用地从像素素组检测
    println!("===test_2===");
    let img = image::open("./test/test_2.png").unwrap();
    let mut img_array = img.to_rgb8();
    // OpenCV 常用 BGR 格式，这里用 RGB 一般也不影响
    let img_mat =
        OcrUtils::create_mat_from_bgr8(img.height() as i32, img.width() as i32, img_array.as_mut());
    let res = ocr.detect(&img_mat, 50, 1024, 0.5, 0.3, 1.6, true, false)?;
    println!("res: {}", res);

    Ok(())
}
```

Output:

```bash
===test_1===
res: TextBlock[BoxPointsLen(4), BoxScore(0.883992), AngleIndex(0), AngleScore(0.99999976), Text(后续可能会将他作为crate发布，因为目前只是研发在Rust调用PaddleOCR，代码没有整理的很好。实际项目使用后有反馈再做打算。), TextScore(0.9849159)]TextBlock[BoxPointsLen(4), BoxScore(0.85958225), AngleIndex(0), AngleScore(0.89264715), Text(一个尝试通过 Rust 调用 PaddleOCR 实现图片文字提取的测试程序。), TextScore(0.9740863)]TextBlock[BoxPointsLen(4), BoxScore(0.85514736), AngleIndex(0), AngleScore(0.9953818), Text(paddle-ocr-rs), TextScore(0.99551016)]
===test_2===
res: TextBlock[BoxPointsLen(4), BoxScore(0.9588546), AngleIndex(0), AngleScore(0.99999917), Text(母婴用品连锁), TextScore(0.9974614)]
```

#### Development Environment

| Dependency | Version |
|------------|-----------------------------|
| rustc | 1.84.1 (e71f9a9a9 2025-01-27) |
| cargo | 1.84.1 (66221abde 2024-11-19) |
| OpenCV | 4.11.0 |
| OS | Windows 11 24H2 |
| Paddle OCR | 4 |

#### Model Source

[RapidOCR Docs](https://rapidai.github.io/RapidOCRDocs/main/model_list/)

#### Important Notes

This project can be considered as a Rust implementation of RapidOCR, with code referenced from RapidOCR's C++ implementation.

C++ implementation: [RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx)

One tricky point is that the project uses OpenCV for image processing, so an OpenCV environment is required.

The OpenCV crate used is [opencv-rust](https://github.com/twistedfall/opencv-rust), and the environment dependencies installation is described in the README: [INSTALL.md](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md)

Note that you need to place the DLL named opencv_worldxxxx.dll in the ./target/debug directory. This is mentioned in the opencv-rust documentation, so we won't elaborate further here.

#### Demo Results

#### test_1.png

![test_1](./test/test_1.png)

```bash
TextBlock[BoxPointsLen(4), BoxScore(0.883992), AngleIndex(0), AngleScore(0.99999976), Text(后续可能会将他作为crate发布，因为目前只是研发在Rust调用PaddleOCR，代码没有整理的很好。实际项目使用后有反馈再做打算。), TextScore(0.9849159)]TextBlock[BoxPointsLen(4), BoxScore(0.85958225), AngleIndex(0), AngleScore(0.89264715), Text(一个尝试通过 Rust 调用 PaddleOCR 实现图片文字提取的测试程序。), TextScore(0.9740863)]TextBlock[BoxPointsLen(4), BoxScore(0.85514736), AngleIndex(0), AngleScore(0.9953818), Text(paddle-ocr-rs), TextScore(0.99551016)]
```

#### test_2.png

![test_2](./test/test_2.png)

```bash
TextBlock[BoxPointsLen(4), BoxScore(0.9570672), AngleIndex(0), AngleScore(0.9999999), Text(母婴用品连锁), TextScore(0.99932665)]
```

#### Output Preview

```bash
===test_1===
paddle-ocr-rs
一个尝试通过 Rust 调用 PaddleOCR 实现图片文字提取的测试程序。
后续可能会将他作为crate发布，因为目前只是研发在Rust调用PaddleOCR，代码没有整理的很好。实际项目使用后有反馈再做打算。
===test_2===
母婴用品连锁
```
