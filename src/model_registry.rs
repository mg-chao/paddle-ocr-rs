use std::collections::HashMap;

use serde::Deserialize;

use crate::{
    config::{LangCls, LangDet, LangRec, ModelType, OcrVersion},
    error::{PaddleOcrError, Result},
};

const DEFAULT_MODELS_YAML: &str = include_str!("../assets/default_models.yaml");

#[derive(Debug, Clone, Deserialize)]
struct Root {
    onnxruntime: HashMap<String, OcrVersionNode>,
}

#[derive(Debug, Clone, Deserialize)]
struct OcrVersionNode {
    #[serde(default)]
    det: HashMap<String, ModelEntry>,
    #[serde(default)]
    cls: HashMap<String, ModelEntry>,
    #[serde(default)]
    rec: HashMap<String, ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelEntry {
    model_dir: String,
    #[serde(rename = "SHA256")]
    sha256: Option<String>,
    dict_url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelVariant {
    Server,
    Mobile,
}

#[derive(Debug, Clone, Copy)]
struct ModelCandidate<'a> {
    name: &'a String,
    entry: &'a ModelEntry,
    variant: ModelVariant,
}

#[derive(Debug, Clone)]
pub struct ResolvedRecModel {
    pub model_name: String,
    pub model_url: String,
    pub sha256: Option<String>,
    pub dict_url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedTaskModel {
    pub model_name: String,
    pub model_url: String,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelRegistry {
    root: Root,
}

impl ModelRegistry {
    pub fn from_default_yaml() -> Result<Self> {
        Self::from_yaml_str(DEFAULT_MODELS_YAML)
    }

    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        let root = serde_yaml::from_str::<Root>(yaml)?;
        Ok(Self { root })
    }

    pub fn resolve_rec(
        &self,
        ocr_version: OcrVersion,
        lang: LangRec,
        model_type: ModelType,
    ) -> Result<ResolvedRecModel> {
        let version_map = self
            .root
            .onnxruntime
            .get(ocr_version.as_str())
            .ok_or_else(|| {
                PaddleOcrError::ModelResolve(format!(
                    "unsupported ocr version for onnxruntime: {}",
                    ocr_version.as_str()
                ))
            })?;

        let lang_prefix = lang.as_str();
        let selected = select_model(
            &version_map.rec,
            lang_prefix,
            model_type,
            "rec",
            ocr_version,
        )?;

        Ok(ResolvedRecModel {
            model_name: selected.0.clone(),
            model_url: selected.1.model_dir.clone(),
            sha256: selected.1.sha256.clone(),
            dict_url: selected.1.dict_url.clone(),
        })
    }

    pub fn resolve_det(
        &self,
        ocr_version: OcrVersion,
        lang: LangDet,
        model_type: ModelType,
    ) -> Result<ResolvedTaskModel> {
        let version_map = self.version_node(ocr_version)?;
        let selected = select_model(
            &version_map.det,
            lang.as_str(),
            model_type,
            "det",
            ocr_version,
        )?;
        Ok(ResolvedTaskModel {
            model_name: selected.0.clone(),
            model_url: selected.1.model_dir.clone(),
            sha256: selected.1.sha256.clone(),
        })
    }

    pub fn resolve_cls(
        &self,
        ocr_version: OcrVersion,
        lang: LangCls,
        model_type: ModelType,
    ) -> Result<ResolvedTaskModel> {
        let version_map = self.version_node(ocr_version)?;
        let selected = select_model(
            &version_map.cls,
            lang.as_str(),
            model_type,
            "cls",
            ocr_version,
        )?;
        Ok(ResolvedTaskModel {
            model_name: selected.0.clone(),
            model_url: selected.1.model_dir.clone(),
            sha256: selected.1.sha256.clone(),
        })
    }

    fn version_node(&self, ocr_version: OcrVersion) -> Result<&OcrVersionNode> {
        self.root
            .onnxruntime
            .get(ocr_version.as_str())
            .ok_or_else(|| {
                PaddleOcrError::ModelResolve(format!(
                    "unsupported ocr version for onnxruntime: {}",
                    ocr_version.as_str()
                ))
            })
    }
}

fn select_model<'a>(
    model_map: &'a HashMap<String, ModelEntry>,
    lang_prefix: &str,
    model_type: ModelType,
    task: &str,
    ocr_version: OcrVersion,
) -> Result<(&'a String, &'a ModelEntry)> {
    let mut candidates: Vec<ModelCandidate<'a>> = model_map
        .iter()
        .filter(|(name, _)| language_tag_matches(name, lang_prefix))
        .map(|(name, entry)| ModelCandidate {
            name,
            entry,
            variant: classify_model_variant(name),
        })
        .collect();
    candidates.sort_by(|a, b| a.name.cmp(b.name));

    if candidates.is_empty() {
        return Err(PaddleOcrError::ModelResolve(format!(
            "no {task} model found for lang={lang_prefix}, version={}",
            ocr_version.as_str()
        )));
    }

    let selected = match model_type {
        ModelType::Server => select_unique_variant(
            &candidates,
            ModelVariant::Server,
            task,
            lang_prefix,
            ocr_version,
        )?,
        ModelType::Mobile => select_unique_variant(
            &candidates,
            ModelVariant::Mobile,
            task,
            lang_prefix,
            ocr_version,
        )?,
    };
    Ok((selected.name, selected.entry))
}

fn select_unique_variant<'a>(
    candidates: &[ModelCandidate<'a>],
    variant: ModelVariant,
    task: &str,
    lang_prefix: &str,
    ocr_version: OcrVersion,
) -> Result<ModelCandidate<'a>> {
    let matched: Vec<ModelCandidate<'a>> = candidates
        .iter()
        .copied()
        .filter(|candidate| candidate.variant == variant)
        .collect();
    if matched.is_empty() {
        return Err(PaddleOcrError::ModelResolve(format!(
            "no {variant:?} {task} model found for lang={lang_prefix}, version={}",
            ocr_version.as_str()
        )));
    }
    if matched.len() > 1 {
        return Err(PaddleOcrError::ModelResolve(format!(
            "ambiguous {variant:?} {task} models for lang={lang_prefix}, version={}: {}",
            ocr_version.as_str(),
            format_candidate_names(&matched)
        )));
    }
    Ok(matched[0])
}

fn language_tag_matches(model_name: &str, lang_tag: &str) -> bool {
    extract_language_tag(model_name).is_some_and(|candidate| candidate == lang_tag)
}

fn extract_language_tag(model_name: &str) -> Option<&str> {
    let markers = ["_PP-", "_PP", "_ppocr_", "_ppocr"];
    let idx = markers
        .iter()
        .filter_map(|marker| model_name.find(marker))
        .min()?;
    if idx == 0 {
        return None;
    }
    Some(&model_name[..idx])
}

fn classify_model_variant(model_name: &str) -> ModelVariant {
    let lower = model_name.to_ascii_lowercase();
    if lower.contains("_server_") || lower.contains("_server.") {
        return ModelVariant::Server;
    }
    ModelVariant::Mobile
}

fn format_candidate_names(candidates: &[ModelCandidate<'_>]) -> String {
    let mut names: Vec<&str> = candidates
        .iter()
        .map(|candidate| candidate.name.as_str())
        .collect();
    names.sort_unstable();
    names.join(", ")
}

#[cfg(test)]
mod tests {
    use super::ModelRegistry;
    use crate::config::{LangCls, LangDet, LangRec, ModelType, OcrVersion};

    const CUSTOM_YAML: &str = r#"
onnxruntime:
  PP-OCRv4:
    rec:
      ch_PP-OCRv4_rec_infer.onnx:
        model_dir: https://example.com/ch-mobile.onnx
      ch_doc_PP-OCRv4_rec_server_infer.onnx:
        model_dir: https://example.com/ch-doc-server.onnx
      ch_PP-OCRv4_rec_server_infer.onnx:
        model_dir: https://example.com/ch-server.onnx
"#;

    const AMBIGUOUS_MOBILE_YAML: &str = r#"
onnxruntime:
  PP-OCRv4:
    rec:
      en_PP-OCRv4_rec_infer.onnx:
        model_dir: https://example.com/en-mobile-a.onnx
      en_PP-OCRv4_rec_infer_v2.onnx:
        model_dir: https://example.com/en-mobile-b.onnx
"#;

    #[test]
    fn resolve_server_and_mobile() {
        let reg = ModelRegistry::from_default_yaml().expect("registry should parse");

        let mobile = reg
            .resolve_rec(OcrVersion::PPocrV4, LangRec::Ch, ModelType::Mobile)
            .expect("mobile model should resolve");
        assert!(mobile.model_name.contains("ch_PP-OCRv4_rec_infer"));
        assert!(!mobile.model_name.contains("server"));

        let server = reg
            .resolve_rec(OcrVersion::PPocrV4, LangRec::Ch, ModelType::Server)
            .expect("server model should resolve");
        assert!(server.model_name.contains("server"));
    }

    #[test]
    fn resolve_det_and_cls() {
        let reg = ModelRegistry::from_default_yaml().expect("registry should parse");

        let det = reg
            .resolve_det(OcrVersion::PPocrV4, LangDet::Ch, ModelType::Mobile)
            .expect("det model should resolve");
        assert!(det.model_name.contains("det"));

        let cls = reg
            .resolve_cls(OcrVersion::PPocrV4, LangCls::Ch, ModelType::Mobile)
            .expect("cls model should resolve");
        assert!(cls.model_name.contains("cls"));
    }

    #[test]
    fn resolve_lang_prefix_match_is_exact() {
        let reg = ModelRegistry::from_yaml_str(CUSTOM_YAML).expect("registry should parse");
        let mobile = reg
            .resolve_rec(OcrVersion::PPocrV4, LangRec::Ch, ModelType::Mobile)
            .expect("mobile model should resolve");
        assert!(mobile.model_name.starts_with("ch_PP-OCRv4_rec_infer"));
    }

    #[test]
    fn resolve_server_requires_explicit_server_variant() {
        let reg = ModelRegistry::from_default_yaml().expect("registry should parse");
        let err = reg
            .resolve_rec(OcrVersion::PPocrV4, LangRec::En, ModelType::Server)
            .expect_err("server should require explicit server variant");
        assert!(err.to_string().contains("no Server rec model found"));
    }

    #[test]
    fn resolve_mobile_rejects_ambiguous_mobile_models() {
        let reg =
            ModelRegistry::from_yaml_str(AMBIGUOUS_MOBILE_YAML).expect("registry should parse");
        let err = reg
            .resolve_rec(OcrVersion::PPocrV4, LangRec::En, ModelType::Mobile)
            .expect_err("ambiguous mobile models should fail");
        assert!(err.to_string().contains("ambiguous Mobile rec models"));
    }
}
