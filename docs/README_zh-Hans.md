## paddle-ocr-rs

使用 Rust 通过 ONNX Runtime 调用 Paddle OCR 模型进行图片文字识别。

#### 示例

```rust
use paddle_ocr_rs::{ocr_error::OcrError, ocr_lite::OcrLite};

fn main() -> Result<(), OcrError> {
    let mut ocr = OcrLite::new();
    ocr.init_models(
        "./models/ch_PP-OCRv4_det_infer.onnx",
        "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "./models/ch_PP-OCRv4_rec_infer.onnx",
        "./models/ppocr_keys_v1.txt",
        2,
    )?;

    println!("===test_1===");
    let res = ocr.detect_from_path("./test/test_1.png", 50, 1024, 0.5, 0.3, 1.6, true, false)?;
    println!("res: {}", res);
    println!("===test_2===");
    let res = ocr.detect_from_path("./test/test_2.png", 50, 1024, 0.5, 0.3, 1.6, true, false)?;
    println!("res: {}", res);
    Ok(())
}
```

输出：

```bash
===test_1===
res: TextBlock[BoxPointsLen(4), BoxScore(0.883992), AngleIndex(0), AngleScore(0.99999976), Text(后续可能会将他作为crate发布，因为目前只是研发在Rust调用PaddleOCR，代码没有整理的很好。实际项目使用后有反馈再做打算。), TextScore(0.9849159)]TextBlock[BoxPointsLen(4), BoxScore(0.85958225), AngleIndex(0), AngleScore(0.89264715), Text(一个尝试通过 Rust 调用 PaddleOCR 实现图片文字提取的测试程序。), TextScore(0.9740863)]TextBlock[BoxPointsLen(4), BoxScore(0.85514736), AngleIndex(0), AngleScore(0.9953818), Text(paddle-ocr-rs), TextScore(0.99551016)]
===test_2===
res: TextBlock[BoxPointsLen(4), BoxScore(0.9570672), AngleIndex(0), AngleScore(0.9999999), Text(母婴用品连锁), TextScore(0.99932665)]
```

#### 开发环境

| 依赖 | 版本号 |
|------------|-----------------------------|
| rustc | 1.84.1 (e71f9a9a9 2025-01-27) |
| cargo | 1.84.1 (66221abde 2024-11-19) |
| OpenCV | 4.11.0 |
| OS | Windows 11 24H2 |
| Paddle OCR | 4 |

#### 模型来源

[RapidOCR Docs](https://rapidai.github.io/RapidOCRDocs/main/model_list/)

#### 相关事项

项目可以认为是 RapidOCR 的 Rust 实现，代码参考自 RapidOCR 的 C++ 实现。

C++ 实现：[RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx)

有个比较坑的点是项目用到了 OpenCV 做图片相关的处理，所以需要提供 OpenCV 环境。

OpenCV 的 crate 是 [opencv-rust](https://github.com/twistedfall/opencv-rust)，环境依赖安装在 README 上有介绍：[INSTALL.md](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md)

注意得把名为 opencv_worldxxxx.dll 的 dll 放在 ./target/debug 目录下。opencv-rust 的文档都有提及，在此就不赘叙了。

#### 效果展示

#### test_1.png

![test_1](../test/test_1.png)

```bash
TextBlock[BoxPointsLen(4), BoxScore(0.883992), AngleIndex(0), AngleScore(0.99999976), Text(后续可能会将他作为crate发布，因为目前只是研发在Rust调用PaddleOCR，代码没有整理的很好。实际项目使用后有反馈再做打算。), TextScore(0.9849159)]TextBlock[BoxPointsLen(4), BoxScore(0.85958225), AngleIndex(0), AngleScore(0.89264715), Text(一个尝试通过 Rust 调用 PaddleOCR 实现图片文字提取的测试程序。), TextScore(0.9740863)]TextBlock[BoxPointsLen(4), BoxScore(0.85514736), AngleIndex(0), AngleScore(0.9953818), Text(paddle-ocr-rs), TextScore(0.99551016)]
```

#### test_2.png

![test_2](../test/test_2.png)

```bash
TextBlock[BoxPointsLen(4), BoxScore(0.9570672), AngleIndex(0), AngleScore(0.9999999), Text(母婴用品连锁), TextScore(0.99932665)]
```

#### 输出预览

```bash
===test_1===
paddle-ocr-rs
一个尝试通过 Rust 调用 PaddleOCR 实现图片文字提取的测试程序。
后续可能会将他作为crate发布，因为目前只是研发在Rust调用PaddleOCR，代码没有整理的很好。实际项目使用后有反馈再做打算。
===test_2===
母婴用品连锁
```

