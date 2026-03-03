use std::{fs, path::Path};

use serde_yaml::Value;

use crate::{
    config::ProviderPreference,
    error::{PaddleOcrError, Result},
    pipeline::config::EngineConfig,
};

mod convert;
mod schema;

use convert::{
    mapping_get, parse_lang_cls, parse_lang_det, parse_lang_rec, parse_model_type,
    parse_ocr_version, value_to_bool, value_to_f32, value_to_f32x3, value_to_pathbuf_string,
    value_to_string, value_to_string_vec, value_to_usize, value_to_usizex3,
};
use schema::OnnxRuntimeCompat;

pub fn from_rapidocr_yaml_str(yaml: &str) -> Result<EngineConfig> {
    let root = serde_yaml::from_str::<Value>(yaml)?;
    let mut cfg = EngineConfig::default();

    apply_global_section(&root, &mut cfg);
    apply_det_section(&root, &mut cfg)?;
    apply_cls_section(&root, &mut cfg)?;
    apply_rec_section(&root, &mut cfg)?;
    apply_onnxruntime_section(&root, &mut cfg);

    cfg.validate()?;
    Ok(cfg)
}

pub fn from_rapidocr_yaml_file(path: impl AsRef<Path>) -> Result<EngineConfig> {
    let text = fs::read_to_string(path)?;
    from_rapidocr_yaml_str(&text)
}

fn apply_global_section(root: &Value, cfg: &mut EngineConfig) {
    let Some(global) = mapping_get(root, "Global") else {
        return;
    };

    if let Some(v) = mapping_get(global, "text_score").and_then(value_to_f32) {
        cfg.global.text_score = v;
    }
    if let Some(v) = mapping_get(global, "use_det").and_then(value_to_bool) {
        cfg.global.use_det = v;
    }
    if let Some(v) = mapping_get(global, "use_cls").and_then(value_to_bool) {
        cfg.global.use_cls = v;
    }
    if let Some(v) = mapping_get(global, "use_rec").and_then(value_to_bool) {
        cfg.global.use_rec = v;
    }
    if let Some(v) = mapping_get(global, "min_height").and_then(value_to_usize) {
        cfg.global.min_height = v;
    }
    if let Some(v) = mapping_get(global, "width_height_ratio").and_then(value_to_f32) {
        cfg.global.width_height_ratio = v;
    }
    if let Some(v) = mapping_get(global, "max_side_len").and_then(value_to_usize) {
        cfg.global.max_side_len = v;
    }
    if let Some(v) = mapping_get(global, "min_side_len").and_then(value_to_usize) {
        cfg.global.min_side_len = v;
    }
    if let Some(v) = mapping_get(global, "return_word_box").and_then(value_to_bool) {
        cfg.global.return_word_box = v;
    }
    if let Some(v) = mapping_get(global, "return_single_char_box").and_then(value_to_bool) {
        cfg.global.return_single_char_box = v;
    }
}

fn apply_det_section(root: &Value, cfg: &mut EngineConfig) -> Result<()> {
    let Some(det) = mapping_get(root, "Det") else {
        return Ok(());
    };

    validate_engine_type(det, "Det")?;

    if let Some(v) = mapping_get(det, "lang_type")
        .and_then(value_to_string)
        .and_then(parse_lang_det)
    {
        cfg.det.lang = v;
    }
    if let Some(v) = mapping_get(det, "ocr_version")
        .and_then(value_to_string)
        .and_then(parse_ocr_version)
    {
        cfg.det.ocr_version = v;
    }
    if let Some(v) = mapping_get(det, "model_type")
        .and_then(value_to_string)
        .and_then(parse_model_type)
    {
        cfg.det.model_type = v;
    }
    if let Some(v) = mapping_get(det, "model_path").and_then(value_to_pathbuf_string) {
        cfg.det.model_path = Some(v.into());
    }
    if let Some(v) = mapping_get(det, "limit_side_len").and_then(value_to_usize) {
        cfg.det.limit_side_len = v;
    }
    if let Some(v) = mapping_get(det, "limit_type").and_then(value_to_string) {
        cfg.det.limit_type = v;
    }
    if let Some(v) = mapping_get(det, "thresh").and_then(value_to_f32) {
        cfg.det.thresh = v;
    }
    if let Some(v) = mapping_get(det, "box_thresh").and_then(value_to_f32) {
        cfg.det.box_thresh = v;
    }
    if let Some(v) = mapping_get(det, "max_candidates").and_then(value_to_usize) {
        cfg.det.max_candidates = v;
    }
    if let Some(v) = mapping_get(det, "unclip_ratio").and_then(value_to_f32) {
        cfg.det.unclip_ratio = v;
    }
    if let Some(v) = mapping_get(det, "use_dilation").and_then(value_to_bool) {
        cfg.det.use_dilation = v;
    }
    if let Some(v) = mapping_get(det, "score_mode").and_then(value_to_string) {
        cfg.det.score_mode = v;
    }
    if let Some(v) = mapping_get(det, "mean").and_then(value_to_f32x3) {
        cfg.det.mean = v;
    }
    if let Some(v) = mapping_get(det, "std").and_then(value_to_f32x3) {
        cfg.det.std = v;
    }

    Ok(())
}

fn apply_cls_section(root: &Value, cfg: &mut EngineConfig) -> Result<()> {
    let Some(cls) = mapping_get(root, "Cls") else {
        return Ok(());
    };

    validate_engine_type(cls, "Cls")?;

    if let Some(v) = mapping_get(cls, "lang_type")
        .and_then(value_to_string)
        .and_then(parse_lang_cls)
    {
        cfg.cls.lang = v;
    }
    if let Some(v) = mapping_get(cls, "ocr_version")
        .and_then(value_to_string)
        .and_then(parse_ocr_version)
    {
        cfg.cls.ocr_version = v;
    }
    if let Some(v) = mapping_get(cls, "model_type")
        .and_then(value_to_string)
        .and_then(parse_model_type)
    {
        cfg.cls.model_type = v;
    }
    if let Some(v) = mapping_get(cls, "model_path").and_then(value_to_pathbuf_string) {
        cfg.cls.model_path = Some(v.into());
    }
    if let Some(v) = mapping_get(cls, "cls_image_shape").and_then(value_to_usizex3) {
        cfg.cls.cls_image_shape = v;
    }
    if let Some(v) = mapping_get(cls, "cls_batch_num").and_then(value_to_usize) {
        cfg.cls.cls_batch_num = v;
    }
    if let Some(v) = mapping_get(cls, "cls_thresh").and_then(value_to_f32) {
        cfg.cls.cls_thresh = v;
    }
    if let Some(v) = mapping_get(cls, "label_list").and_then(value_to_string_vec) {
        cfg.cls.label_list = v;
    }

    Ok(())
}

fn apply_rec_section(root: &Value, cfg: &mut EngineConfig) -> Result<()> {
    let Some(rec) = mapping_get(root, "Rec") else {
        return Ok(());
    };

    validate_engine_type(rec, "Rec")?;

    if let Some(v) = mapping_get(rec, "lang_type")
        .and_then(value_to_string)
        .and_then(parse_lang_rec)
    {
        cfg.rec.model.lang = v;
    }
    if let Some(v) = mapping_get(rec, "ocr_version")
        .and_then(value_to_string)
        .and_then(parse_ocr_version)
    {
        cfg.rec.model.ocr_version = v;
    }
    if let Some(v) = mapping_get(rec, "model_type")
        .and_then(value_to_string)
        .and_then(parse_model_type)
    {
        cfg.rec.model.model_type = v;
    }
    if let Some(v) = mapping_get(rec, "model_path").and_then(value_to_pathbuf_string) {
        cfg.rec.model.model_path = Some(v.into());
    }
    if let Some(v) = mapping_get(rec, "rec_keys_path").and_then(value_to_pathbuf_string) {
        cfg.rec.model.rec_keys_path = Some(v.into());
    }
    if let Some(v) = mapping_get(rec, "rec_batch_num").and_then(value_to_usize) {
        cfg.rec.rec_batch_num = v;
    }
    if let Some(v) = mapping_get(rec, "rec_img_shape").and_then(value_to_usizex3) {
        cfg.rec.rec_img_shape = v;
    }

    Ok(())
}

fn apply_onnxruntime_section(root: &Value, cfg: &mut EngineConfig) {
    let Some(onnx_cfg_value) =
        mapping_get(root, "EngineConfig").and_then(|v| mapping_get(v, "onnxruntime"))
    else {
        return;
    };

    let onnx_cfg =
        serde_yaml::from_value::<OnnxRuntimeCompat>(onnx_cfg_value.clone()).unwrap_or_default();

    let intra = onnx_cfg.intra_op_num_threads;
    let inter = onnx_cfg.inter_op_num_threads;
    let auto_tune_threads = onnx_cfg.auto_tune_threads.unwrap_or(true);
    let rayon_threads = onnx_cfg.rayon_threads;
    let mem_arena = onnx_cfg.enable_cpu_mem_arena.unwrap_or(false);
    let fail_if_provider_unavailable = onnx_cfg.fail_if_provider_unavailable.unwrap_or(false);
    let provider = provider_from_compat_onnxruntime(&onnx_cfg);

    for runtime in [
        &mut cfg.det.runtime,
        &mut cfg.cls.runtime,
        &mut cfg.rec.runtime,
    ] {
        runtime.intra_threads = intra;
        runtime.inter_threads = inter;
        runtime.auto_tune_threads = auto_tune_threads;
        runtime.rayon_threads = rayon_threads;
        runtime.enable_cpu_mem_arena = mem_arena;
        runtime.fail_if_provider_unavailable = fail_if_provider_unavailable;
        runtime.provider_preference = provider;
    }
}

fn validate_engine_type(section: &Value, section_name: &str) -> Result<()> {
    if let Some(engine_type) = mapping_get(section, "engine_type").and_then(value_to_string)
        && engine_type != "onnxruntime"
    {
        return Err(PaddleOcrError::Config(format!(
            "only onnxruntime engine_type is supported in Rust v1, got `{engine_type}` in {section_name}"
        )));
    }
    Ok(())
}

fn provider_from_compat_onnxruntime(cfg: &OnnxRuntimeCompat) -> ProviderPreference {
    if cfg.use_cuda.unwrap_or(false) {
        return ProviderPreference::Cuda {
            device_id: cfg
                .cuda_ep_cfg
                .as_ref()
                .and_then(|v| v.device_id)
                .unwrap_or(0),
        };
    }
    if cfg.use_dml.unwrap_or(false) {
        return ProviderPreference::DirectMl {
            device_id: cfg
                .dm_ep_cfg
                .as_ref()
                .and_then(|v| v.device_id)
                .unwrap_or(0),
        };
    }
    if cfg.use_cann.unwrap_or(false) {
        return ProviderPreference::Cann {
            device_id: cfg
                .cann_ep_cfg
                .as_ref()
                .and_then(|v| v.device_id)
                .unwrap_or(0),
        };
    }
    ProviderPreference::Cpu
}

#[cfg(test)]
mod tests {
    use super::from_rapidocr_yaml_str;
    use crate::config::{ModelType, OcrVersion, ProviderPreference};

    #[test]
    fn parse_rapidocr_yaml_compat() {
        let yaml = r#"
Global:
  text_score: 0.66
  use_det: true
  use_cls: false
  use_rec: true
Det:
  engine_type: onnxruntime
  lang_type: ch
  ocr_version: PP-OCRv4
  model_type: mobile
  box_thresh: 0.42
Cls:
  engine_type: onnxruntime
  lang_type: ch
  ocr_version: PP-OCRv4
  model_type: mobile
Rec:
  engine_type: onnxruntime
  lang_type: en
  ocr_version: PP-OCRv5
  model_type: mobile
EngineConfig:
  onnxruntime:
    intra_op_num_threads: 2
    inter_op_num_threads: 1
    enable_cpu_mem_arena: true
"#;
        let cfg = from_rapidocr_yaml_str(yaml).expect("compat parse should pass");
        assert!((cfg.global.text_score - 0.66).abs() < f32::EPSILON);
        assert!(cfg.global.use_det);
        assert!(!cfg.global.use_cls);
        assert!(cfg.global.use_rec);
        assert!((cfg.det.box_thresh - 0.42).abs() < f32::EPSILON);
        assert_eq!(cfg.rec.model.ocr_version.as_str(), "PP-OCRv5");
        assert_eq!(cfg.rec.runtime.intra_threads, Some(2));
        assert_eq!(cfg.rec.runtime.inter_threads, Some(1));
        assert!(cfg.rec.runtime.enable_cpu_mem_arena);
    }

    #[test]
    fn parse_rapidocr_yaml_rejects_invalid_runtime_values() {
        let yaml = r#"
Det:
  engine_type: onnxruntime
Cls:
  engine_type: onnxruntime
Rec:
  engine_type: onnxruntime
EngineConfig:
  onnxruntime:
    intra_op_num_threads: 0
"#;
        let err = from_rapidocr_yaml_str(yaml).expect_err("zero threads should fail validation");
        assert!(
            err.to_string()
                .contains("det.runtime.intra_threads must be greater than zero when set")
        );
    }

    #[test]
    fn parse_rapidocr_yaml_directml_provider() {
        let yaml = r#"
Det:
  engine_type: onnxruntime
Cls:
  engine_type: onnxruntime
Rec:
  engine_type: onnxruntime
EngineConfig:
  onnxruntime:
    use_dml: true
    dm_ep_cfg:
      device_id: 2
"#;
        let cfg = from_rapidocr_yaml_str(yaml).expect("compat parse should pass");
        assert_eq!(
            cfg.rec.runtime.provider_preference,
            ProviderPreference::DirectMl { device_id: 2 }
        );
        assert_eq!(
            cfg.det.runtime.provider_preference,
            ProviderPreference::DirectMl { device_id: 2 }
        );
        assert_eq!(
            cfg.cls.runtime.provider_preference,
            ProviderPreference::DirectMl { device_id: 2 }
        );
    }

    #[test]
    fn parse_rapidocr_yaml_thread_tuning() {
        let yaml = r#"
Det:
  engine_type: onnxruntime
Cls:
  engine_type: onnxruntime
Rec:
  engine_type: onnxruntime
EngineConfig:
  onnxruntime:
    auto_tune_threads: false
    rayon_threads: 6
"#;
        let cfg = from_rapidocr_yaml_str(yaml).expect("compat parse should pass");
        assert!(!cfg.det.runtime.auto_tune_threads);
        assert_eq!(cfg.det.runtime.rayon_threads, Some(6));
        assert!(!cfg.cls.runtime.auto_tune_threads);
        assert_eq!(cfg.cls.runtime.rayon_threads, Some(6));
        assert!(!cfg.rec.runtime.auto_tune_threads);
        assert_eq!(cfg.rec.runtime.rayon_threads, Some(6));
    }

    #[test]
    fn parse_rapidocr_yaml_provider_strictness_flag() {
        let yaml = r#"
Det:
  engine_type: onnxruntime
Cls:
  engine_type: onnxruntime
Rec:
  engine_type: onnxruntime
EngineConfig:
  onnxruntime:
    fail_if_provider_unavailable: true
"#;
        let cfg = from_rapidocr_yaml_str(yaml).expect("compat parse should pass");
        assert!(cfg.det.runtime.fail_if_provider_unavailable);
        assert!(cfg.cls.runtime.fail_if_provider_unavailable);
        assert!(cfg.rec.runtime.fail_if_provider_unavailable);
    }

    #[test]
    fn parse_rapidocr_yaml_prefers_cuda_over_other_provider_flags() {
        let yaml = r#"
Det:
  engine_type: onnxruntime
Cls:
  engine_type: onnxruntime
Rec:
  engine_type: onnxruntime
EngineConfig:
  onnxruntime:
    use_cuda: true
    use_dml: true
    use_cann: true
    cuda_ep_cfg:
      device_id: 7
    dm_ep_cfg:
      device_id: 3
"#;
        let cfg = from_rapidocr_yaml_str(yaml).expect("compat parse should pass");
        assert_eq!(
            cfg.det.runtime.provider_preference,
            ProviderPreference::Cuda { device_id: 7 }
        );
    }

    #[test]
    fn parse_rapidocr_yaml_treats_typed_runtime_parse_failure_as_defaults() {
        let yaml = r#"
Det:
  engine_type: onnxruntime
Cls:
  engine_type: onnxruntime
Rec:
  engine_type: onnxruntime
EngineConfig:
  onnxruntime:
    intra_op_num_threads: "abc"
"#;
        let cfg = from_rapidocr_yaml_str(yaml).expect("compat parse should pass");
        assert_eq!(cfg.det.runtime.intra_threads, None);
        assert_eq!(cfg.cls.runtime.intra_threads, None);
        assert_eq!(cfg.rec.runtime.intra_threads, None);
    }

    #[test]
    fn parse_rapidocr_yaml_accepts_model_type_aliases_via_converter() {
        let yaml = r#"
Det:
  engine_type: onnxruntime
  model_type: server
Cls:
  engine_type: onnxruntime
Rec:
  engine_type: onnxruntime
"#;
        let cfg = from_rapidocr_yaml_str(yaml).expect("compat parse should pass");
        assert_eq!(cfg.det.model_type, ModelType::Server);
        assert_eq!(cfg.det.ocr_version, OcrVersion::PPocrV4);
    }
}
