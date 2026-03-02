use std::{path::PathBuf, time::Instant};

use ndarray::ArrayView4;
use rayon::prelude::*;

use crate::{
    config::{LangRec, RecImage, RecognizeOptions, RecognizerConfig, VisionBackend},
    error::{PaddleOcrError, Result},
    model_registry::{ModelRegistry, ResolvedRecModel},
    model_store::{default_model_store_dir, ensure_downloaded, verify_existing_file},
    rec::{
        bidi::reorder_bidi_for_display,
        decode::CtcLabelDecoder,
        preprocess::{batch_shape_for, write_resize_norm_img_into_slice_with_scratch},
    },
    runtime::provider::ProviderResolution,
    runtime::session::OrtSession,
    types::{LineResult, RecognizeOutput},
    vision::backend::resolve_backend_strict,
    vision::resize::LinearResizeScratch,
};

#[derive(Debug)]
pub struct Recognizer {
    config: RecognizerConfig,
    vision_backend: VisionBackend,
    session: OrtSession,
    decoder: CtcLabelDecoder,
    batch_scratch: Vec<f32>,
}

impl Recognizer {
    pub fn new(config: RecognizerConfig) -> Result<Self> {
        if config.rec_img_shape[0] != 3 {
            return Err(PaddleOcrError::Config(format!(
                "rec_img_shape must start with channel=3, got {:?}",
                config.rec_img_shape
            )));
        }
        if config.rec_batch_num == 0 {
            return Err(PaddleOcrError::Config(
                "rec_batch_num must be greater than zero".to_string(),
            ));
        }

        let model_store_dir = config
            .model_store_dir
            .clone()
            .unwrap_or_else(default_model_store_dir);
        let vision_backend = resolve_backend_strict(config.runtime.vision_backend)?;

        let registry = ModelRegistry::from_default_yaml()?;
        let resolved = registry.resolve_rec(
            config.model.ocr_version,
            config.model.lang,
            config.model.model_type,
        )?;

        let model_path = resolve_model_path(&config, &resolved, &model_store_dir)?;
        let mut session = OrtSession::new(&model_path, &config.runtime)?;

        let character = session.character_list.take();
        let character_path = if character.is_none() {
            resolve_character_path(&config, &resolved, &model_store_dir)?
        } else {
            config.model.rec_keys_path.clone()
        };

        let decoder = CtcLabelDecoder::new(character, character_path.as_deref())?;

        Ok(Self {
            config,
            vision_backend,
            session,
            decoder,
            batch_scratch: Vec::new(),
        })
    }

    pub fn recognize(
        &mut self,
        images: &[RecImage],
        options: RecognizeOptions,
    ) -> Result<RecognizeOutput> {
        let start = Instant::now();

        if images.is_empty() {
            return Ok(RecognizeOutput::default());
        }

        let width_list: Vec<f64> = images
            .iter()
            .map(|img| img.width() as f64 / img.height() as f64)
            .collect();
        let mut indices: Vec<usize> = (0..images.len()).collect();
        indices.sort_by(|a, b| {
            width_list[*a]
                .partial_cmp(&width_list[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut rec_res: Vec<Option<LineResult>> = vec![None; images.len()];

        for beg in (0..images.len()).step_by(self.config.rec_batch_num) {
            let end = (beg + self.config.rec_batch_num).min(images.len());
            let batch_indices = &indices[beg..end];

            let img_h = self.config.rec_img_shape[1] as f64;
            let img_w = self.config.rec_img_shape[2] as f64;
            let mut max_wh_ratio = img_w / img_h;

            let mut wh_ratio_list = Vec::with_capacity(end - beg);
            for sorted_idx in batch_indices {
                let img = &images[*sorted_idx];
                let wh_ratio = img.width() as f64 / img.height() as f64;
                max_wh_ratio = max_wh_ratio.max(wh_ratio);
                wh_ratio_list.push(wh_ratio as f32);
            }

            let (img_channel, img_height, dst_width) =
                batch_shape_for(max_wh_ratio, self.config.rec_img_shape)?;
            let sample_len = img_channel
                .checked_mul(img_height)
                .and_then(|v| v.checked_mul(dst_width))
                .ok_or_else(|| {
                    PaddleOcrError::InvalidInput("rec batch sample size overflow".to_string())
                })?;
            let batch_size = batch_indices.len();
            let total_len = sample_len.checked_mul(batch_size).ok_or_else(|| {
                PaddleOcrError::InvalidInput("rec batch size overflow".to_string())
            })?;
            self.batch_scratch.resize(total_len, 0.0);

            if batch_size > 1 {
                self.batch_scratch[..total_len]
                    .par_chunks_mut(sample_len)
                    .zip(batch_indices.par_iter().copied())
                    .try_for_each_init(
                        || (Vec::<u8>::new(), LinearResizeScratch::default()),
                        |(tmp_bgr, resize_scratch), (dst, image_idx)| {
                            let image = images.get(image_idx).ok_or_else(|| {
                                PaddleOcrError::InvalidInput(format!(
                                    "batch index {image_idx} out of bounds for image count {}",
                                    images.len()
                                ))
                            })?;
                            write_resize_norm_img_into_slice_with_scratch(
                                image,
                                max_wh_ratio,
                                self.config.rec_img_shape,
                                self.vision_backend,
                                dst,
                                tmp_bgr,
                                resize_scratch,
                            )
                        },
                    )?;
            } else {
                let image_idx = batch_indices[0];
                let image = images.get(image_idx).ok_or_else(|| {
                    PaddleOcrError::InvalidInput(format!(
                        "batch index {image_idx} out of bounds for image count {}",
                        images.len()
                    ))
                })?;
                let mut tmp_bgr = Vec::new();
                let mut resize_scratch = LinearResizeScratch::default();
                write_resize_norm_img_into_slice_with_scratch(
                    image,
                    max_wh_ratio,
                    self.config.rec_img_shape,
                    self.vision_backend,
                    &mut self.batch_scratch[..sample_len],
                    &mut tmp_bgr,
                    &mut resize_scratch,
                )?;
            }

            let batch_view = ArrayView4::from_shape(
                (batch_size, img_channel, img_height, dst_width),
                &self.batch_scratch[..total_len],
            )
            .map_err(|e| {
                PaddleOcrError::InvalidInput(format!("invalid rec batch tensor shape: {e}"))
            })?;
            let decoder = &self.decoder;
            let (line_results, word_results) =
                self.session.run_array3_view_with(batch_view, |preds| {
                    decoder.decode_view(
                        preds,
                        options.return_word_box,
                        &wh_ratio_list,
                        max_wh_ratio as f32,
                    )
                })?;

            for (rno, (text, score)) in line_results.into_iter().enumerate() {
                let word_info = if options.return_word_box {
                    word_results.get(rno).cloned()
                } else {
                    None
                };

                let target_idx = indices[beg + rno];
                rec_res[target_idx] = Some(LineResult {
                    text,
                    score,
                    word_info,
                });
            }
        }

        let mut lines = Vec::with_capacity(images.len());
        for line in rec_res.into_iter().flatten() {
            lines.push(line);
        }

        if self.config.model.lang == LangRec::Arabic {
            for line in &mut lines {
                line.text = reorder_bidi_for_display(&line.text);
            }
        }

        Ok(RecognizeOutput {
            lines,
            elapsed: start.elapsed(),
        })
    }

    pub fn provider_resolution(&self) -> ProviderResolution {
        self.session.provider_resolution()
    }
}

fn resolve_model_path(
    config: &RecognizerConfig,
    resolved: &ResolvedRecModel,
    model_store_dir: &PathBuf,
) -> Result<PathBuf> {
    if let Some(model_path) = &config.model.model_path {
        return verify_existing_file(model_path);
    }

    if !config.model.allow_download {
        return Err(PaddleOcrError::Config(
            "model_path is not set and allow_download=false".to_string(),
        ));
    }

    ensure_downloaded(
        &resolved.model_url,
        resolved.sha256.as_deref(),
        model_store_dir,
    )
}

fn resolve_character_path(
    config: &RecognizerConfig,
    resolved: &ResolvedRecModel,
    model_store_dir: &PathBuf,
) -> Result<Option<PathBuf>> {
    if let Some(path) = &config.model.rec_keys_path {
        return Ok(Some(verify_existing_file(path)?));
    }

    let Some(dict_url) = &resolved.dict_url else {
        return Ok(None);
    };

    if !config.model.allow_download {
        return Err(PaddleOcrError::Config(
            "character metadata missing and dict download disabled".to_string(),
        ));
    }

    let path = ensure_downloaded(dict_url, None, model_store_dir)?;
    Ok(Some(path))
}
