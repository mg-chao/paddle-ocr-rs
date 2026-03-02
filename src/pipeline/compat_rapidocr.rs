use std::{fs, path::Path};

use serde_yaml::Value;

use crate::{
    config::{LangCls, LangDet, LangRec, ModelType, OcrVersion, ProviderPreference},
    error::{PaddleOcrError, Result},
    pipeline::config::EngineConfig,
};

pub fn from_rapidocr_yaml_str(yaml: &str) -> Result<EngineConfig> {
    let root = serde_yaml::from_str::<Value>(yaml)?;
    let mut cfg = EngineConfig::default();

    if let Some(global) = mapping_get(&root, "Global") {
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

    if let Some(det) = mapping_get(&root, "Det") {
        if let Some(engine_type) = mapping_get(det, "engine_type").and_then(value_to_string)
            && engine_type != "onnxruntime"
        {
            return Err(PaddleOcrError::Config(format!(
                "only onnxruntime engine_type is supported in Rust v1, got `{engine_type}` in Det"
            )));
        }
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
    }

    if let Some(cls) = mapping_get(&root, "Cls") {
        if let Some(engine_type) = mapping_get(cls, "engine_type").and_then(value_to_string)
            && engine_type != "onnxruntime"
        {
            return Err(PaddleOcrError::Config(format!(
                "only onnxruntime engine_type is supported in Rust v1, got `{engine_type}` in Cls"
            )));
        }
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
    }

    if let Some(rec) = mapping_get(&root, "Rec") {
        if let Some(engine_type) = mapping_get(rec, "engine_type").and_then(value_to_string)
            && engine_type != "onnxruntime"
        {
            return Err(PaddleOcrError::Config(format!(
                "only onnxruntime engine_type is supported in Rust v1, got `{engine_type}` in Rec"
            )));
        }
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
    }

    if let Some(onnx_cfg) =
        mapping_get(&root, "EngineConfig").and_then(|v| mapping_get(v, "onnxruntime"))
    {
        let intra = mapping_get(onnx_cfg, "intra_op_num_threads").and_then(value_to_usize);
        let inter = mapping_get(onnx_cfg, "inter_op_num_threads").and_then(value_to_usize);
        let auto_tune_threads = mapping_get(onnx_cfg, "auto_tune_threads")
            .and_then(value_to_bool)
            .unwrap_or(true);
        let rayon_threads = mapping_get(onnx_cfg, "rayon_threads").and_then(value_to_usize);
        let mem_arena = mapping_get(onnx_cfg, "enable_cpu_mem_arena")
            .and_then(value_to_bool)
            .unwrap_or(false);
        let fail_if_provider_unavailable = mapping_get(onnx_cfg, "fail_if_provider_unavailable")
            .and_then(value_to_bool)
            .unwrap_or(false);

        let provider = if mapping_get(onnx_cfg, "use_cuda")
            .and_then(value_to_bool)
            .unwrap_or(false)
        {
            let device_id = mapping_get(onnx_cfg, "cuda_ep_cfg")
                .and_then(|v| mapping_get(v, "device_id"))
                .and_then(value_to_usize)
                .unwrap_or(0);
            ProviderPreference::Cuda { device_id }
        } else if mapping_get(onnx_cfg, "use_dml")
            .and_then(value_to_bool)
            .unwrap_or(false)
        {
            let device_id = mapping_get(onnx_cfg, "dm_ep_cfg")
                .and_then(|v| mapping_get(v, "device_id"))
                .and_then(value_to_usize)
                .unwrap_or(0);
            ProviderPreference::DirectMl { device_id }
        } else if mapping_get(onnx_cfg, "use_cann")
            .and_then(value_to_bool)
            .unwrap_or(false)
        {
            let device_id = mapping_get(onnx_cfg, "cann_ep_cfg")
                .and_then(|v| mapping_get(v, "device_id"))
                .and_then(value_to_usize)
                .unwrap_or(0);
            ProviderPreference::Cann { device_id }
        } else {
            ProviderPreference::Cpu
        };

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

    cfg.validate()?;
    Ok(cfg)
}

pub fn from_rapidocr_yaml_file(path: impl AsRef<Path>) -> Result<EngineConfig> {
    let text = fs::read_to_string(path)?;
    from_rapidocr_yaml_str(&text)
}

fn mapping_get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_mapping()?.get(Value::String(key.to_string()))
}

fn value_to_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(v) => Some(*v),
        Value::Number(v) => v.as_i64().map(|x| x != 0),
        Value::String(v) => match v.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Some(true),
            "false" | "0" | "no" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn value_to_f32(value: &Value) -> Option<f32> {
    match value {
        Value::Number(v) => v.as_f64().map(|x| x as f32),
        Value::String(v) => v.parse::<f32>().ok(),
        _ => None,
    }
}

fn value_to_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(v) => v.as_i64(),
        Value::String(v) => v.parse::<i64>().ok(),
        _ => None,
    }
}

fn value_to_usize(value: &Value) -> Option<usize> {
    value_to_i64(value).filter(|v| *v >= 0).map(|v| v as usize)
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(v) => Some(v.clone()),
        _ => None,
    }
}

fn value_to_pathbuf_string(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(v) if v.trim().is_empty() => None,
        Value::String(v) => Some(v.clone()),
        _ => None,
    }
}

fn value_to_f32x3(value: &Value) -> Option<[f32; 3]> {
    let seq = value.as_sequence()?;
    if seq.len() != 3 {
        return None;
    }
    Some([
        value_to_f32(&seq[0])?,
        value_to_f32(&seq[1])?,
        value_to_f32(&seq[2])?,
    ])
}

fn value_to_usizex3(value: &Value) -> Option<[usize; 3]> {
    let seq = value.as_sequence()?;
    if seq.len() != 3 {
        return None;
    }
    Some([
        value_to_usize(&seq[0])?,
        value_to_usize(&seq[1])?,
        value_to_usize(&seq[2])?,
    ])
}

fn value_to_string_vec(value: &Value) -> Option<Vec<String>> {
    let seq = value.as_sequence()?;
    let mut out = Vec::with_capacity(seq.len());
    for item in seq {
        out.push(value_to_string(item)?);
    }
    Some(out)
}

fn parse_lang_det(value: String) -> Option<LangDet> {
    match value.as_str() {
        "ch" => Some(LangDet::Ch),
        "en" => Some(LangDet::En),
        "multi" => Some(LangDet::Multi),
        _ => None,
    }
}

fn parse_lang_cls(value: String) -> Option<LangCls> {
    match value.as_str() {
        "ch" => Some(LangCls::Ch),
        _ => None,
    }
}

fn parse_lang_rec(value: String) -> Option<LangRec> {
    match value.as_str() {
        "ch" => Some(LangRec::Ch),
        "ch_doc" => Some(LangRec::ChDoc),
        "en" => Some(LangRec::En),
        "arabic" => Some(LangRec::Arabic),
        "chinese_cht" => Some(LangRec::ChineseCht),
        "cyrillic" => Some(LangRec::Cyrillic),
        "devanagari" => Some(LangRec::Devanagari),
        "japan" => Some(LangRec::Japan),
        "korean" => Some(LangRec::Korean),
        "ka" => Some(LangRec::Ka),
        "latin" => Some(LangRec::Latin),
        "ta" => Some(LangRec::Ta),
        "te" => Some(LangRec::Te),
        "eslav" => Some(LangRec::Eslav),
        "th" => Some(LangRec::Th),
        "el" => Some(LangRec::El),
        _ => None,
    }
}

fn parse_model_type(value: String) -> Option<ModelType> {
    match value.as_str() {
        "mobile" => Some(ModelType::Mobile),
        "server" => Some(ModelType::Server),
        _ => None,
    }
}

fn parse_ocr_version(value: String) -> Option<OcrVersion> {
    match value.as_str() {
        "PP-OCRv4" => Some(OcrVersion::PPocrV4),
        "PP-OCRv5" => Some(OcrVersion::PPocrV5),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::from_rapidocr_yaml_str;
    use crate::config::ProviderPreference;

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
}
