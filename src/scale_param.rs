use opencv::core::Mat;
use opencv::prelude::MatTraitConst;

#[derive(Debug)]
pub struct ScaleParam {
    pub src_width: i32,
    pub src_height: i32,
    pub dst_width: i32,
    pub dst_height: i32,
    pub scale_width: f32,
    pub scale_height: f32,
}

impl ScaleParam {
    pub fn new(
        src_width: i32,
        src_height: i32,
        dst_width: i32,
        dst_height: i32,
        scale_width: f32,
        scale_height: f32,
    ) -> Self {
        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            scale_width,
            scale_height,
        }
    }

    // Getters
    pub fn src_width(&self) -> i32 { self.src_width }
    pub fn src_height(&self) -> i32 { self.src_height }
    pub fn dst_width(&self) -> i32 { self.dst_width }
    pub fn dst_height(&self) -> i32 { self.dst_height }
    pub fn scale_width(&self) -> f32 { self.scale_width }
    pub fn scale_height(&self) -> f32 { self.scale_height }

    pub fn get_scale_param(src: &Mat, dst_size: i32) -> Self {
        let src_width = src.cols();
        let mut dst_width = src.cols();
        let src_height = src.rows();
        let mut dst_height = src.rows();

        let scale;
        if dst_width > dst_height {
            scale = dst_size as f32 / dst_width as f32;
            dst_width = dst_size;
            dst_height = (dst_height as f32 * scale) as i32;
        } else {
            scale = dst_size as f32 / dst_height as f32;
            dst_height = dst_size;
            dst_width = (dst_width as f32 * scale) as i32;
        }

        if dst_width % 32 != 0 {
            dst_width = (dst_width / 32 - 1) * 32;
            dst_width = dst_width.max(32);
        }
        if dst_height % 32 != 0 {
            dst_height = (dst_height / 32 - 1) * 32;
            dst_height = dst_height.max(32);
        }

        let scale_width = dst_width as f32 / src_width as f32;
        let scale_height = dst_height as f32 / src_height as f32;

        Self::new(
            src_width,
            src_height,
            dst_width,
            dst_height,
            scale_width,
            scale_height,
        )
    }
}

impl std::fmt::Display for ScaleParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sw:{},sh:{},dw:{},dh:{},{},{}",
            self.src_width, self.src_height, self.dst_width, self.dst_height,
            self.scale_width, self.scale_height
        )
    }
}