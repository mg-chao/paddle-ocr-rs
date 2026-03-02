pub mod cls;
pub mod config;
pub mod det;
pub mod error;
pub mod input;
pub mod model_registry;
pub mod model_store;
pub mod output;
pub mod pipeline;
pub mod rec;
pub mod runtime;
pub mod types;
pub mod vision;

pub use config::{
    ColorOrder, LangCls, LangDet, LangRec, ModelConfig, ModelType, OcrVersion, ProviderPreference,
    RecImage, RecognizeOptions, RecognizerConfig, RuntimeBackend, RuntimeConfig, VisionBackend,
};
pub use error::{PaddleOcrError, Result};
pub use input::image_loader::{LoadImage, OcrInput};
pub use output::json::OcrJsonItem;
pub use pipeline::{
    config::{EngineConfig, GlobalConfig},
    rapid_ocr::{PipelineProviderResolutions, RapidOcr, RapidOcrEngine},
    types::{
        ClsResult, DetResult, FullResult, OcrCallOptions, OcrOutput, OcrResult, RecResult,
        RunOptions, StageTimings,
    },
};
pub use rec::recognizer::Recognizer;
pub use rec::word_boxes::{compute_word_boxes, compute_word_boxes_with_backend};
pub use runtime::provider::{ProviderResolution, ResolvedExecutionProvider};
pub use types::{LineResult, RecognizeOutput, WordBox, WordInfo, WordType};
pub use vision::{OcrTaskVisionPolicy, VisionBackendPolicy};

pub type Quad = [[f32; 2]; 4];
