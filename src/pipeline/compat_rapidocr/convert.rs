use serde_yaml::Value;

use crate::config::{LangCls, LangDet, LangRec, ModelType, OcrVersion};

pub(crate) fn mapping_get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_mapping()?.get(Value::String(key.to_string()))
}

pub(crate) fn value_to_bool(value: &Value) -> Option<bool> {
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

pub(crate) fn value_to_f32(value: &Value) -> Option<f32> {
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

pub(crate) fn value_to_usize(value: &Value) -> Option<usize> {
    value_to_i64(value).filter(|v| *v >= 0).map(|v| v as usize)
}

pub(crate) fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(v) => Some(v.clone()),
        _ => None,
    }
}

pub(crate) fn value_to_pathbuf_string(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(v) if v.trim().is_empty() => None,
        Value::String(v) => Some(v.clone()),
        _ => None,
    }
}

pub(crate) fn value_to_f32x3(value: &Value) -> Option<[f32; 3]> {
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

pub(crate) fn value_to_usizex3(value: &Value) -> Option<[usize; 3]> {
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

pub(crate) fn value_to_string_vec(value: &Value) -> Option<Vec<String>> {
    let seq = value.as_sequence()?;
    let mut out = Vec::with_capacity(seq.len());
    for item in seq {
        out.push(value_to_string(item)?);
    }
    Some(out)
}

pub(crate) fn parse_lang_det(value: String) -> Option<LangDet> {
    match value.as_str() {
        "ch" => Some(LangDet::Ch),
        "en" => Some(LangDet::En),
        "multi" => Some(LangDet::Multi),
        _ => None,
    }
}

pub(crate) fn parse_lang_cls(value: String) -> Option<LangCls> {
    match value.as_str() {
        "ch" => Some(LangCls::Ch),
        _ => None,
    }
}

pub(crate) fn parse_lang_rec(value: String) -> Option<LangRec> {
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

pub(crate) fn parse_model_type(value: String) -> Option<ModelType> {
    match value.as_str() {
        "mobile" => Some(ModelType::Mobile),
        "server" => Some(ModelType::Server),
        _ => None,
    }
}

pub(crate) fn parse_ocr_version(value: String) -> Option<OcrVersion> {
    match value.as_str() {
        "PP-OCRv4" => Some(OcrVersion::PPocrV4),
        "PP-OCRv5" => Some(OcrVersion::PPocrV5),
        _ => None,
    }
}
