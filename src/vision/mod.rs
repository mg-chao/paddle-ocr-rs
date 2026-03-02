pub(crate) mod backend;
pub(crate) mod image_backend;
pub(crate) mod resize;
pub(crate) mod rotate_crop;

use crate::config::VisionBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OcrTaskVisionPolicy {
    pub rec_backend: VisionBackend,
    pub det_backend: VisionBackend,
    pub cls_backend: VisionBackend,
}

pub trait VisionBackendPolicy {
    fn backend_for_rec(&self) -> VisionBackend;
    fn backend_for_det(&self) -> VisionBackend;
    fn backend_for_cls(&self) -> VisionBackend;
}

impl VisionBackendPolicy for OcrTaskVisionPolicy {
    fn backend_for_rec(&self) -> VisionBackend {
        self.rec_backend
    }

    fn backend_for_det(&self) -> VisionBackend {
        self.det_backend
    }

    fn backend_for_cls(&self) -> VisionBackend {
        self.cls_backend
    }
}

#[cfg(test)]
mod tests {
    use super::{OcrTaskVisionPolicy, VisionBackendPolicy};
    use crate::config::VisionBackend;

    #[test]
    fn policy_defaults_are_consistent() {
        let p = OcrTaskVisionPolicy::default();
        assert_eq!(p.backend_for_rec(), VisionBackend::default());
        assert_eq!(p.backend_for_det(), VisionBackend::default());
        assert_eq!(p.backend_for_cls(), VisionBackend::default());
    }
}
