use crate::ocr_result::{Point, TextBox};
use image::{ImageBuffer, Rgb};
use ndarray::{Array, Array4};
use opencv::{
    core::{Mat, Point as CvPoint, Scalar, Size, Vector},
    imgproc::{get_perspective_transform, warp_perspective},
    prelude::*,
};

pub struct OcrUtils;

impl OcrUtils {
    pub fn substract_mean_normalize(
        src: &Mat,
        mean_vals: &[f32],
        norm_vals: &[f32],
    ) -> Array4<f32> {
        let cols = src.cols();
        let rows = src.rows();
        let channels = src.channels();

        let mut input_tensor = Array::zeros((1, channels as usize, rows as usize, cols as usize));

        // 获取图像数据
        unsafe {
            let data = src.data();
            for r in 0..rows {
                for c in 0..cols {
                    for ch in 0..channels {
                        let idx = (r * cols * channels + c * channels + ch) as usize;
                        let value = *data.add(idx);
                        let data = value as f32 * norm_vals[ch as usize]
                            - mean_vals[ch as usize] * norm_vals[ch as usize];
                        input_tensor[[0, ch as usize, r as usize, c as usize]] = data;
                    }
                }
            }
        }

        input_tensor
    }

    pub fn make_padding(src: &Mat, padding: i32) -> Mat {
        if padding <= 0 {
            return src.clone();
        }
        let padding_scalar = Scalar::new(255.0, 255.0, 255.0, 0.0);
        let mut padding_src = Mat::default();
        opencv::core::copy_make_border(
            src,
            &mut padding_src,
            padding,
            padding,
            padding,
            padding,
            opencv::core::BORDER_ISOLATED,
            padding_scalar,
        )
        .expect("Failed to make padding");
        padding_src
    }

    pub fn get_thickness(box_img: &Mat) -> i32 {
        let min_size = box_img.cols().min(box_img.rows());
        min_size / 1000 + 2
    }

    pub fn draw_text_box(box_img: &mut Mat, box_points: &[Point], thickness: i32) {
        if box_points.is_empty() {
            return;
        }
        let color = Scalar::new(0.0, 0.0, 255.0, 0.0); // B(0) G(0) R(255)

        for i in 0..4 {
            let start = CvPoint::new(box_points[i].x, box_points[i].y);
            let end = CvPoint::new(box_points[(i + 1) % 4].x, box_points[(i + 1) % 4].y);
            opencv::imgproc::line(
                box_img,
                start,
                end,
                color,
                thickness,
                opencv::imgproc::LINE_8,
                0,
            )
            .expect("Failed to draw line");
        }
    }

    pub fn draw_text_boxes(src: &mut Mat, text_boxes: &[TextBox], thickness: i32) {
        for text_box in text_boxes {
            Self::draw_text_box(src, &text_box.points, thickness);
        }
    }

    pub fn get_part_images(src: &Mat, text_boxes: &[TextBox]) -> Vec<Mat> {
        text_boxes
            .iter()
            .map(|text_box| Self::get_rotate_crop_image(src, &text_box.points))
            .collect()
    }

    pub fn get_rotate_crop_image(src: &Mat, box_points: &[Point]) -> Mat {
        let mut points = box_points.to_vec();

        // 计算边界框
        let x_coords: Vec<i32> = points.iter().map(|p| p.x).collect();
        let y_coords: Vec<i32> = points.iter().map(|p| p.y).collect();
        let left = *x_coords.iter().min().unwrap();
        let right = *x_coords.iter().max().unwrap();
        let top = *y_coords.iter().min().unwrap();
        let bottom = *y_coords.iter().max().unwrap();

        // 裁剪图像
        let rect = opencv::core::Rect::new(left, top, right - left, bottom - top);
        let img_crop = Mat::roi(src, rect).unwrap();

        // 调整点坐标
        for point in &mut points {
            point.x -= left;
            point.y -= top;
        }

        // 计算目标图像尺寸
        let img_crop_width = ((points[0].x - points[1].x).pow(2) as f32
            + (points[0].y - points[1].y).pow(2) as f32)
            .sqrt() as i32;
        let img_crop_height = ((points[0].x - points[3].x).pow(2) as f32
            + (points[0].y - points[3].y).pow(2) as f32)
            .sqrt() as i32;

        // 创建源点和目标点
        let binding = [
            points[0].x as f32,
            points[0].y as f32,
            points[1].x as f32,
            points[1].y as f32,
            points[2].x as f32,
            points[2].y as f32,
            points[3].x as f32,
            points[3].y as f32,
        ];
        let binding = Mat::from_slice(&binding).unwrap();
        let pts_src = binding.reshape(2, 4).unwrap();

        let binding = [
            0.0f32,
            0.0f32,
            img_crop_width as f32,
            0.0f32,
            img_crop_width as f32,
            img_crop_height as f32,
            0.0f32,
            img_crop_height as f32,
        ];
        let binding = Mat::from_slice(&binding).unwrap();
        let pts_dst = binding.reshape(2, 4).unwrap();

        // 获取透视变换矩阵
        let m = get_perspective_transform(&pts_src, &pts_dst, opencv::core::DECOMP_LU)
            .expect("Failed to get perspective transform");

        // 进行透视变换
        let mut part_img = Mat::default();
        warp_perspective(
            &img_crop,
            &mut part_img,
            &m,
            Size::new(img_crop_width, img_crop_height),
            opencv::imgproc::INTER_NEAREST,
            opencv::core::BORDER_REPLICATE,
            Scalar::default(),
        )
        .expect("Failed to warp perspective");

        // 根据需要旋转图像
        if part_img.rows() >= part_img.cols() * 3 / 2 {
            let mut src_copy = Mat::default();
            opencv::core::transpose(&part_img, &mut src_copy).expect("Failed to transpose");
            opencv::core::flip(&src_copy.clone(), &mut src_copy, 0).expect("Failed to flip");
            src_copy
        } else {
            part_img
        }
    }

    pub fn mat_rotate_clock_wise_180(src: &mut Mat) {
        opencv::core::flip(&src.clone(), src, -1).expect("Failed to flip");
    }

    pub fn mat_rotate_clock_wise_90(src: &mut Mat) {
        opencv::core::rotate(&src.clone(), src, opencv::core::ROTATE_90_COUNTERCLOCKWISE)
            .expect("Failed to rotate");
    }

    // 添加一个新函数，用于将预测数据保存为图像文件
    fn save_pred_data_to_image(
        r_pred_data: &Vector<u8>,
        g_pred_data: &Vector<u8>,
        b_pred_data: &Vector<u8>,
        rows: i32,
        cols: i32,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 创建一个新的图像缓冲区
        let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(cols as u32, rows as u32);

        for x in 0..cols {
            for y in 0..rows {
                let idx = (y * cols + x) as usize;
                let r = r_pred_data.get(idx).unwrap();
                let g = g_pred_data.get(idx).unwrap();
                let b = b_pred_data.get(idx).unwrap();
                img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }
        // 保存图像
        img.save(filename)?;
        println!("已保存预测结果到文件: {}", filename);

        Ok(())
    }
}
