## paddle-ocr-rs

一个尝试通过 Rust 调用 PaddleOCR 实现图片文字提取的测试程序。

后续可能会将他作为 crate 发布，因为目前只是研发在 Rust 调用 PaddleOCR，代码没有整理的很好。实际项目使用后有反馈再做打算。

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
paddle-ocr-rs
~个尝试通过Rust 调用PaddleOCR 实现图片文字提取的测试程序。
后续可能会将他作为 crate 发布，因为目前只是研发在 Rust 调用 PaddleOCR，代码没有整理的很好。实际项目使用后有反馈再做打算。
```

#### test_2.png

![test_2](../test/test_2.png)

```bash
母婴用品连
锁
```

#### 输出预览

##### test_1.png

```bash
keys Size = 6625
=====Start detect=====
---------- step: dbNet getTextBoxes ----------
TextBoxesSize(3)
TextBox { points: [Point { x: 102, y: 131 }, Point { x: 334, y: 177 }, Point { x: 327, y: 219 }, Point { x: 94, y: 174 }], score: 0.8739496 }
TextBox { points: [Point { x: 90, y: 197 }, Point { x: 796, y: 333 }, Point { x: 790, y: 369 }, Point { x: 84, y: 233 }], score: 0.8564408 }
TextBox { points: [Point { x: 77, y: 257 }, Point { x: 1478, y: 527 }, Point { x: 1472, y: 564 }, Point { x: 70, y: 294 }], score: 0.8536175 }
---------- step: drawTextBoxes ----------
---------- step: angleNet getAngles ----------
AnglesSize(3)
Angle { index: 0, score: 1.0, time: 15.0 }
Angle { index: 0, score: 0.6775721, time: 14.0 }
Angle { index: 0, score: 0.9986754, time: 13.0 }
---------- step: crnnNet getTextLines ----------
paddle-ocr-rs
~个尝试通过Rust 调用PaddleOCR 实现图片文字提取的测试程序。
后续可能会将他作为 crate 发布，因为目前只是研发在 Rust 调用 PaddleOCR，代码没有整理的很好。实际项目使用后有反馈再做打算。
```

