use crate::{
    ocr_error::OcrError,
    ocr_result::{self, TextBox},
    ocr_utils::OcrUtils,
    scale_param::ScaleParam,
};
use geo_clipper::{Clipper, EndType, JoinType};
use geo_types::{Coord, LineString, Polygon};
use opencv::{
    core::{Mat, MatTraitConst, Point, Point2f, RotatedRect, Scalar, Size, Vector, BORDER_DEFAULT},
    imgproc::{self},
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::{inputs, session::SessionOutputs};
use std::cmp::Ordering;

const MEAN_VALUES: [f32; 3] = [
    0.485_f32 * 255_f32,
    0.456_f32 * 255_f32,
    0.406_f32 * 255_f32,
];
const NORM_VALUES: [f32; 3] = [
    1.0_f32 / 0.229_f32 / 255.0_f32,
    1.0_f32 / 0.224_f32 / 255.0_f32,
    1.0_f32 / 0.225_f32 / 255.0_f32,
];

#[derive(Debug)]
pub struct DbNet {
    session: Option<Session>,
    input_names: Vec<String>,
}

impl DbNet {
    pub fn new() -> Self {
        Self {
            session: None,
            input_names: Vec::new(),
        }
    }

    pub fn init_model(&mut self, path: &str, num_thread: usize) -> Result<(), OcrError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level2)?
            .with_intra_threads(num_thread)?
            .with_inter_threads(num_thread)?
            .commit_from_file(path)?;

        // Get input names
        let input_names: Vec<String> = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        self.input_names = input_names;
        self.session = Some(session);

        Ok(())
    }

    pub fn get_text_boxes(
        &self,
        src: &Mat,
        scale: &ScaleParam,
        box_score_thresh: f32,
        box_thresh: f32,
        un_clip_ratio: f32,
    ) -> Result<Vec<TextBox>, OcrError> {
        let Some(session) = &self.session else {
            return Err(OcrError::SessionNotInitialized);
        };

        let mut src_resize = Mat::default();
        imgproc::resize(
            &src,
            &mut src_resize,
            Size::new(scale.dst_width, scale.dst_height),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let input_tensors =
            OcrUtils::substract_mean_normalize(&src_resize, &MEAN_VALUES, &NORM_VALUES);

        let outputs = session.run(inputs![self.input_names[0].clone() => input_tensors]?)?;

        let text_boxes = Self::get_text_boxes_core(
            &outputs,
            src_resize.rows(),
            src_resize.cols(),
            scale,
            box_score_thresh,
            box_thresh,
            un_clip_ratio,
        )?;

        Ok(text_boxes)
    }

    fn get_text_boxes_core(
        output_tensor: &SessionOutputs,
        rows: i32,
        cols: i32,
        s: &ScaleParam,
        box_score_thresh: f32,
        box_thresh: f32,
        un_clip_ratio: f32,
    ) -> Result<Vec<TextBox>, OcrError> {
        let max_side_thresh = 3.0;
        let mut rs_boxes = Vec::new();

        let (_, red_data) = output_tensor.iter().next().unwrap();

        let mut pred_data: Vector<f32> = red_data
            .try_extract_tensor::<f32>()?
            .iter()
            .map(|&x| x)
            .collect();

        let mut cbuf_data: Vector<u8> = pred_data
            .iter()
            .map(|pixel| (pixel * 255.0) as u8)
            .collect();

        let pred_mat = OcrUtils::create_mat_from_array(
            rows,
            cols,
            opencv::core::CV_32FC1,
            pred_data.as_mut_slice(),
            0,
        );

        let cbuf_mat = OcrUtils::create_mat_from_array(
            rows,
            cols,
            opencv::core::CV_8UC1,
            cbuf_data.as_mut_slice(),
            0,
        );

        let mut threshold_mat = Mat::default();
        imgproc::threshold(
            &cbuf_mat,
            &mut threshold_mat,
            (box_thresh * 255.0) as f64,
            255.0,
            imgproc::THRESH_BINARY,
        )?;

        let mut dilate_mat = Mat::default();
        let dilate_element = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            Size::new(2, 2),
            Point::new(-1, -1),
        )?;

        imgproc::dilate(
            &threshold_mat,
            &mut dilate_mat,
            &dilate_element,
            Point::new(-1, -1),
            1,
            BORDER_DEFAULT,
            Scalar::all(128.0),
        )?;

        // Find contours
        let mut contours = Vector::<Vector<Point>>::new();
        imgproc::find_contours(
            &dilate_mat,
            &mut contours,
            imgproc::RETR_LIST as i32,
            imgproc::CHAIN_APPROX_SIMPLE as i32,
            Point::new(0, 0),
        )?;

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            if contour.len() <= 2 {
                continue;
            }

            let mut max_side = 0.0;
            let min_box = Self::get_mini_box(&contour, &mut max_side)?;
            if max_side < max_side_thresh {
                continue;
            }

            let score = Self::get_score(&contour, &pred_mat)?;
            if score < box_score_thresh as f64 {
                continue;
            }

            let clip_box = Self::unclip(&min_box, un_clip_ratio)?;
            if clip_box.is_empty() {
                continue;
            }

            let mut clip_contour = Vector::<Point>::new();
            for point in &clip_box {
                clip_contour.push(*point);
            }

            let mut max_side_clip = 0.0;
            let clip_min_box = Self::get_mini_box(&clip_contour, &mut max_side_clip)?;
            if max_side_clip < max_side_thresh + 2.0 {
                continue;
            }

            let mut final_points = Vec::new();
            for item in clip_min_box {
                let x = (item.x / s.scale_width) as i32;
                let ptx = x.max(0).min(s.src_width);

                let y = (item.y / s.scale_height) as i32;
                let pty = y.max(0).min(s.src_height);

                final_points.push(ocr_result::Point { x: ptx, y: pty });
            }

            let text_box = TextBox {
                score: score as f32,
                points: final_points,
            };

            rs_boxes.push(text_box);
        }

        rs_boxes.reverse();
        Ok(rs_boxes)
    }

    fn get_mini_box(
        contour: &Vector<Point>,
        min_edge_size: &mut f32,
    ) -> Result<Vec<Point2f>, OcrError> {
        let rrect: RotatedRect = imgproc::min_area_rect(&contour)?;

        let mut points = [Point2f::default(); 4];
        rrect.points(&mut points)?;

        *min_edge_size = rrect.size.width.min(rrect.size.height);

        let mut the_points: Vec<Point2f> = points.into_iter().collect();
        the_points.sort_by(|a, b| {
            if a.x > b.x {
                return Ordering::Greater;
            }
            if a.x == b.x {
                return Ordering::Equal;
            }
            Ordering::Less
        });

        let mut box_points = Vec::new();
        let index_1;
        let index_4;
        if the_points[1].y > the_points[0].y {
            index_1 = 0;
            index_4 = 1;
        } else {
            index_1 = 1;
            index_4 = 0;
        }

        let index_2;
        let index_3;
        if the_points[3].y > the_points[2].y {
            index_2 = 2;
            index_3 = 3;
        } else {
            index_2 = 3;
            index_3 = 2;
        }

        box_points.push(the_points[index_1]);
        box_points.push(the_points[index_2]);
        box_points.push(the_points[index_3]);
        box_points.push(the_points[index_4]);

        Ok(box_points)
    }

    fn get_score(contour: &Vector<Point>, f_map_mat: &Mat) -> Result<f64, OcrError> {
        // 初始化边界值
        let mut xmin = i32::MAX;
        let mut xmax = i32::MIN;
        let mut ymin = i32::MAX;
        let mut ymax = i32::MIN;

        // 找到轮廓的边界框
        for point in contour {
            let x = point.x as i32;
            let y = point.y as i32;

            if x < xmin {
                xmin = x;
            }
            if x > xmax {
                xmax = x;
            }
            if y < ymin {
                ymin = y;
            }
            if y > ymax {
                ymax = y;
            }
        }

        let width = f_map_mat.cols();
        let height = f_map_mat.rows();

        xmin = xmin.max(0).min(width - 1);
        xmax = xmax.max(0).min(width - 1);
        ymin = ymin.max(0).min(height - 1);
        ymax = ymax.max(0).min(height - 1);

        let roi_width = xmax - xmin + 1;
        let roi_height = ymax - ymin + 1;

        if roi_width <= 0 || roi_height <= 0 {
            return Ok(0.0);
        }

        let mut mask = Mat::new_rows_cols_with_default(
            roi_height,
            roi_width,
            opencv::core::CV_8UC1,
            Scalar::all(0.0),
        )?;

        let mut pts = Vector::<Point>::new();
        for point in contour {
            pts.push(Point::new((point.x as i32) - xmin, (point.y as i32) - ymin));
        }

        let mut vpp_array = Vector::<Vector<Point>>::new();
        vpp_array.push(pts);

        imgproc::fill_poly(
            &mut mask,
            &vpp_array,
            Scalar::from(1.0),
            imgproc::LINE_8,
            0,
            Point::default(),
        )?;

        let roi = opencv::core::Rect::new(xmin, ymin, roi_width, roi_height);
        let cropped_img = f_map_mat.roi(roi)?;

        let mean = opencv::core::mean(&cropped_img, &mask)?;

        Ok(mean[0])
    }

    fn unclip(box_points: &[Point2f], unclip_ratio: f32) -> Result<Vec<Point>, OcrError> {
        let mut points_arr = Vector::<Point2f>::new();
        for pt in box_points {
            points_arr.push(*pt);
        }

        let clip_rect = imgproc::min_area_rect(&points_arr)?;
        if clip_rect.size.height < 1.001 && clip_rect.size.width < 1.001 {
            return Ok(Vec::new());
        }

        let mut the_cliper_pts = Vec::new();
        for pt in box_points {
            let a1 = Coord {
                x: pt.x as f64,
                y: pt.y as f64,
            };
            the_cliper_pts.push(a1);
        }

        let area = Self::signed_polygon_area(box_points).abs();
        let length = Self::length_of_points(box_points);
        let distance = area * unclip_ratio as f32 / length as f32;

        let co = Polygon::new(LineString::new(the_cliper_pts), vec![]);
        let solution = co
            .offset(
                distance as f64,
                JoinType::Round(2.0),
                EndType::ClosedPolygon,
                1.0,
            )
            .0;

        if solution.len() == 0 {
            return Ok(Vec::new());
        }

        let mut ret_pts = Vec::new();
        for ip in solution.first().unwrap().exterior().points() {
            ret_pts.push(Point::new(ip.x() as i32, ip.y() as i32));
        }

        Ok(ret_pts)
    }

    fn signed_polygon_area(points: &[Point2f]) -> f32 {
        let num_points = points.len();
        let mut pts = Vec::with_capacity(num_points + 1);
        pts.extend_from_slice(points);
        pts.push(points[0]);

        let mut area = 0.0;
        for i in 0..num_points {
            area += (pts[i + 1].x - pts[i].x) * (pts[i + 1].y + pts[i].y) / 2.0;
        }

        area
    }

    fn length_of_points(box_points: &[Point2f]) -> f64 {
        if box_points.is_empty() {
            return 0.0;
        }

        let mut length = 0.0;
        let pt = box_points[0];
        let mut x0 = pt.x as f64;
        let mut y0 = pt.y as f64;

        let mut box_with_first = Vec::from(box_points);
        box_with_first.push(pt);

        for idx in 1..box_with_first.len() {
            let pts = box_with_first[idx];
            let x1 = pts.x as f64;
            let y1 = pts.y as f64;
            let dx = x1 - x0;
            let dy = y1 - y0;

            length += (dx * dx + dy * dy).sqrt();

            x0 = x1;
            y0 = y1;
        }

        length
    }
}
