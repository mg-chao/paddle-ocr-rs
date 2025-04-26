use std::ffi::c_void;

use crate::{
    ocr_error::OcrError,
    ocr_result::{Point, PointV2, TextBox, TextBoxV2},
};
use image::imageops;
use imageproc::geometric_transformations::{Interpolation, Projection};
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

    pub fn create_mat_from_bgr32f(rows: i32, cols: i32, array: &mut [f32]) -> Mat {
        Self::create_mat_from_array(rows, cols, opencv::core::CV_32FC3, array, 0)
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

    pub fn substract_mean_normalize_v2(
        img_src: &image::RgbImage,
        mean_vals: &[f32],
        norm_vals: &[f32],
    ) -> Array4<f32> {
        let cols = img_src.width();
        let rows = img_src.height();
        let channels = 3;

        let mut input_tensor = Array::zeros((1, channels as usize, rows as usize, cols as usize));

        // 获取图像数据
        unsafe {
            for r in 0..rows {
                for c in 0..cols {
                    for ch in 0..channels {
                        let idx = (r * cols * channels + c * channels + ch) as usize;
                        let value = img_src.get_unchecked(idx).to_owned();
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

    pub fn make_padding_v2(
        img_src: &image::RgbImage,
        padding: u32,
    ) -> Result<image::RgbImage, OcrError> {
        if padding == 0 {
            return Ok(img_src.clone());
        }

        let width = img_src.width();
        let height = img_src.height();

        let mut padding_src = image::RgbImage::new(width + 2 * padding, height + 2 * padding);
        imageproc::drawing::draw_filled_rect_mut(
            &mut padding_src,
            imageproc::rect::Rect::at(0, 0).of_size(width + 2 * padding, height + 2 * padding),
            image::Rgb([255, 255, 255]),
        );

        image::imageops::replace(&mut padding_src, img_src, padding as i64, padding as i64);

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

    pub fn get_part_images_v2(
        img_src: &mut image::RgbImage,
        text_boxes: &[TextBoxV2],
    ) -> Vec<image::RgbImage> {
        text_boxes
            .iter()
            .map(|text_box| Self::get_rotate_crop_image_v2(img_src, &text_box.points))
            .collect()
    }

    pub fn get_rotate_crop_image(src: &Mat, box_points: &[Point]) -> Mat {
        let mut points = box_points.to_vec();

        let x_coords: Vec<i32> = points.iter().map(|p| p.x).collect();
        let y_coords: Vec<i32> = points.iter().map(|p| p.y).collect();
        let left = *x_coords.iter().min().unwrap();
        let right = *x_coords.iter().max().unwrap();
        let top = *y_coords.iter().min().unwrap();
        let bottom = *y_coords.iter().max().unwrap();

        let rect = opencv::core::Rect::new(left, top, right - left, bottom - top);
        let img_crop = Mat::roi(src, rect).unwrap();

        for point in &mut points {
            point.x -= left;
            point.y -= top;
        }

        let img_crop_width = ((points[0].x - points[1].x).pow(2) as f32
            + (points[0].y - points[1].y).pow(2) as f32)
            .sqrt() as i32;
        let img_crop_height = ((points[0].x - points[3].x).pow(2) as f32
            + (points[0].y - points[3].y).pow(2) as f32)
            .sqrt() as i32;

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

        let m = get_perspective_transform(&pts_src, &pts_dst, opencv::core::DECOMP_LU)
            .expect("Failed to get perspective transform");

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

        if part_img.rows() >= part_img.cols() * 3 / 2 {
            let mut src_copy = Mat::default();
            opencv::core::transpose(&part_img, &mut src_copy).expect("Failed to transpose");
            opencv::core::flip(&src_copy.clone(), &mut src_copy, 0).expect("Failed to flip");
            src_copy
        } else {
            part_img
        }
    }

    pub fn get_rotate_crop_image_v2(
        img_src: &mut image::RgbImage,
        box_points: &[PointV2],
    ) -> image::RgbImage {
        let mut points = box_points.to_vec();

        // 计算边界框
        let (min_x, min_y, max_x, max_y) = points.iter().fold(
            (u32::MAX, u32::MAX, 0u32, 0u32),
            |(min_x, min_y, max_x, max_y), point| {
                (
                    min_x.min(point.x),
                    min_y.min(point.y),
                    max_x.max(point.x),
                    max_y.max(point.y),
                )
            },
        );

        // 裁剪图像
        let img_crop =
            imageops::crop_imm(img_src, min_x, min_y, max_x - min_x, max_y - min_y).to_image();

        let mut writer = std::fs::File::create(format!(
            "./test_output/part_img_img_crop_{}_{}.png",
            img_crop.width(),
            img_crop.height()
        ))
        .unwrap();
        img_crop
            .write_to(&mut writer, image::ImageFormat::Png)
            .unwrap();

        for point in &mut points {
            point.x -= min_x;
            point.y -= min_y;
        }

        let img_crop_width = ((points[0].x as i32 - points[1].x as i32).pow(2) as f32
            + (points[0].y as i32 - points[1].y as i32).pow(2) as f32)
            .sqrt() as u32;
        let img_crop_height = ((points[0].x as i32 - points[3].x as i32).pow(2) as f32
            + (points[0].y as i32 - points[3].y as i32).pow(2) as f32)
            .sqrt() as u32;

        let src_points = [
            (points[0].x as f32, points[0].y as f32),
            (points[1].x as f32, points[1].y as f32),
            (points[2].x as f32, points[2].y as f32),
            (points[3].x as f32, points[3].y as f32),
        ];

        let dst_points = [
            (0.0, 0.0),
            (img_crop_width as f32, 0.0),
            (img_crop_width as f32, img_crop_height as f32),
            (0.0, img_crop_height as f32),
        ];

        let projection = Projection::from_control_points(src_points, dst_points)
            .expect("Failed to create projection transformation");

        let mut part_img = image::RgbImage::new(img_crop_width, img_crop_height);
        imageproc::geometric_transformations::warp_into(
            &img_crop,
            &projection,
            Interpolation::Nearest,
            image::Rgb([255, 255, 255]),
            &mut part_img,
        );

        // 根据需要旋转图像
        if part_img.height() >= part_img.width() * 3 / 2 {
            let mut rotated = image::RgbImage::new(part_img.height(), part_img.width());

            for (x, y, pixel) in part_img.enumerate_pixels() {
                rotated.put_pixel(y, part_img.width() - 1 - x, *pixel);
            }

            rotated
        } else {
            part_img
        }
    }

    pub fn mat_rotate_clock_wise_180(src: &mut Mat) {
        opencv::core::flip(&src.clone(), src, -1).expect("Failed to flip");
    }

    pub fn mat_rotate_clock_wise_180_v2(src: &mut image::RgbImage) {
        imageops::rotate180_in_place(src);
    }

    pub fn calculate_mean_with_mask(
        img: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
        mask: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>,
    ) -> f32 {
        let mut sum: f32 = 0.0;
        let mut mask_count = 0;

        assert_eq!(img.width(), mask.width());
        assert_eq!(img.height(), mask.height());

        for y in 0..img.height() {
            for x in 0..img.width() {
                let mask_value = mask.get_pixel(x, y)[0];
                if mask_value > 0 {
                    let pixel = img.get_pixel(x, y);
                    sum += pixel[0];
                    mask_count += 1;
                }
            }
        }

        if mask_count == 0 {
            return 0.0;
        }

        sum / mask_count as f32
    }
}
