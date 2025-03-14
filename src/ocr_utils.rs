use std::ffi::c_void;

use crate::{
    ocr_error::OcrError,
    ocr_result::{Point, TextBox},
};
use ndarray::{Array, Array4};
use opencv::{
    core::{Mat, Scalar, Size},
    imgproc::{get_perspective_transform, warp_perspective},
    prelude::*,
};

pub struct OcrUtils;

impl OcrUtils {
    /// These are various constructors that form a matrix. As noted in the AutomaticAllocation, often
    /// the default constructor is enough, and the proper matrix will be allocated by an OpenCV function.
    /// The constructed matrix can further be assigned to another matrix or matrix expression or can be
    /// allocated with Mat::create . In the former case, the old content is de-referenced.
    ///
    /// ## Overloaded parameters
    ///
    /// ## Parameters
    /// * rows: Number of rows in a 2D array.
    /// * cols: Number of columns in a 2D array.
    /// * data: Pointer to the user data. Matrix constructors that take data and step parameters do not
    /// allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    /// data, which means that no data is copied. This operation is very efficient and can be used to
    /// process external data using OpenCV functions. The external data is not automatically deallocated, so
    /// you should take care of it.
    /// * step: Number of bytes each matrix row occupies. The value should include the padding bytes at
    /// the end of each row, if any. If the parameter is missing (set to AUTO_STEP ), no padding is assumed
    /// and the actual step is calculated as cols*elemSize(). See Mat::elemSize.
    ///
    /// ## C++ default parameters
    /// * step: AUTO_STEP
    pub fn create_mat_from_array<T: DataType>(
        rows: i32,
        cols: i32,
        typ: i32,
        array: &mut [T],
        step: usize,
    ) -> Mat {
        unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                rows,
                cols,
                typ,
                array.as_mut_ptr().cast::<c_void>(),
                step,
            )
            .unwrap()
        }
    }

    pub fn create_mat_from_bgr8(rows: i32, cols: i32, array: &mut [u8]) -> Mat {
        Self::create_mat_from_array(rows, cols, opencv::core::CV_8UC3, array, 0)
    }

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

    pub fn make_padding(src: &Mat, padding: i32) -> Result<Mat, OcrError> {
        if padding <= 0 {
            return Ok(src.clone());
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
        )?;

        Ok(padding_src)
    }

    pub fn get_thickness(box_img: &Mat) -> i32 {
        let min_size = box_img.cols().min(box_img.rows());
        min_size / 1000 + 2
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
}
