use std::time::Instant;
use std::{sync::Once, thread};

use crate::{
    cls::classifier::{Classifier, ClassifierConfig},
    config::RecognizeOptions,
    det::detector::{Detector, DetectorConfig},
    error::Result,
    input::image_loader::{LoadImage, OcrInput},
    pipeline::{
        config::EngineConfig,
        image_ops::{
            PreprocessRecord, apply_vertical_padding, crop_text_regions, map_boxes_to_original,
            map_img_to_original, resize_image_within_bounds,
        },
        types::{OcrCallOptions, OcrOutput, OcrResult, RunOptions},
    },
    rec::recognizer::Recognizer,
    runtime::provider::ProviderResolution,
    types::{LineResult, WordBox},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineProviderResolutions {
    pub det: ProviderResolution,
    pub cls: ProviderResolution,
    pub rec: ProviderResolution,
}

#[derive(Debug)]
pub struct RapidOcr {
    config: EngineConfig,
    detector: Detector,
    classifier: Classifier,
    recognizer: Recognizer,
    loader: LoadImage,
}

impl RapidOcr {
    pub fn new(config: EngineConfig) -> Result<Self> {
        init_rayon_global_pool(&config);
        let det = Detector::new(detector_cfg_from_pipeline(&config))?;
        let cls = Classifier::new(classifier_cfg_from_pipeline(&config))?;
        let rec = Recognizer::new(config.rec.clone())?;
        Ok(Self {
            config,
            detector: det,
            classifier: cls,
            recognizer: rec,
            loader: LoadImage,
        })
    }

    pub fn run(&mut self, input: OcrInput, opts: OcrCallOptions) -> Result<OcrOutput> {
        let e2e_start = Instant::now();
        let mut output = OcrOutput::default();

        let use_det = opts.use_det.unwrap_or(self.config.global.use_det);
        let use_cls = opts.use_cls.unwrap_or(self.config.global.use_cls);
        let use_rec = opts.use_rec.unwrap_or(self.config.global.use_rec);
        let need_stage_images = use_cls || use_rec;
        let return_word_box = opts
            .return_word_box
            .unwrap_or(self.config.global.return_word_box);
        let return_single_char_box = opts
            .return_single_char_box
            .unwrap_or(self.config.global.return_single_char_box);
        let text_score = opts.text_score.unwrap_or(self.config.global.text_score);

        let ori_img = self.loader.load(input)?;
        let ori_h = ori_img.height();
        let ori_w = ori_img.width();
        let preprocessing_backend = if use_det {
            self.config.det.runtime.vision_backend
        } else {
            self.config.rec.runtime.vision_backend
        };
        let (mut proc_img, ratio_h, ratio_w) = resize_image_within_bounds(
            ori_img,
            self.config.global.min_side_len,
            self.config.global.max_side_len,
            preprocessing_backend,
        )?;

        let mut preprocess_record = PreprocessRecord {
            ratio_h,
            ratio_w,
            ..PreprocessRecord::default()
        };

        let mut det_boxes = Vec::new();
        let mut det_scores = Vec::new();
        let mut stage_images = Vec::new();

        if use_det {
            let (padded, pad_top) = apply_vertical_padding(
                proc_img,
                self.config.global.width_height_ratio,
                self.config.global.min_height,
            )?;
            proc_img = padded;
            preprocess_record.pad_top = pad_top;

            self.detector
                .update_postprocess(opts.box_thresh, opts.unclip_ratio);
            let det_out = self.detector.detect(&proc_img)?;
            if det_out.boxes.is_empty() {
                output.e2e_ms = Some(e2e_start.elapsed().as_secs_f32() * 1000.0);
                return Ok(output);
            }
            output.elapsed_ms[0] = Some(det_out.elapsed_ms);
            output.det_breakdown_ms = det_out.breakdown;
            det_boxes = det_out.boxes;
            det_scores = det_out.scores;
            if need_stage_images {
                stage_images = crop_text_regions(
                    &proc_img,
                    &det_boxes,
                    self.config.det.runtime.vision_backend,
                )?;
            }
        } else if need_stage_images {
            stage_images.push(proc_img);
        }

        if use_cls {
            let cls_result = self.classifier.classify_in_place(&mut stage_images)?;
            output.cls_res = Some(cls_result.cls_res);
            output.elapsed_ms[1] = Some(cls_result.elapsed_ms);
        }

        let mut lines: Vec<LineResult> = Vec::new();
        if use_rec {
            let rec = self.recognizer.recognize(
                &stage_images,
                RecognizeOptions {
                    return_word_box,
                    return_single_char_box,
                },
            )?;
            output.elapsed_ms[2] = Some(rec.elapsed.as_secs_f32() * 1000.0);
            lines = rec.lines;
        }

        if use_det {
            let mut mapped_boxes = det_boxes;
            map_boxes_to_original(&mut mapped_boxes, preprocess_record, ori_h, ori_w);

            if use_rec {
                let (filtered_boxes, filtered_scores, filtered_lines, kept_indices) =
                    filter_empty_lines_boxes_and_scores(mapped_boxes, det_scores, lines);
                lines = filtered_lines;

                let mut computed_word_boxes = None;
                if return_word_box && !filtered_boxes.is_empty() && !lines.is_empty() {
                    let mapped_crops = map_img_to_original(
                        &stage_images,
                        ratio_h,
                        ratio_w,
                        self.config.det.runtime.vision_backend,
                    )?;
                    let filtered_crops = select_items_by_indices(mapped_crops, &kept_indices);
                    let word_boxes = crate::rec::word_boxes::compute_word_boxes_with_backend(
                        &filtered_crops,
                        &filtered_boxes,
                        &lines,
                        return_single_char_box,
                        self.config.rec.runtime.vision_backend,
                    )?;
                    computed_word_boxes = Some(word_boxes);
                }

                let (
                    score_filtered_boxes,
                    score_filtered_scores,
                    score_filtered_lines,
                    score_filtered_words,
                ) = filter_by_text_score_for_full(
                    filtered_boxes,
                    filtered_scores,
                    lines,
                    computed_word_boxes,
                    text_score,
                );

                lines = score_filtered_lines;
                output.boxes = Some(score_filtered_boxes);
                output.det_scores = Some(score_filtered_scores);
                output.word_boxes = score_filtered_words;
            } else {
                output.boxes = Some(mapped_boxes);
                output.det_scores = Some(det_scores);
            }
        }

        if use_rec {
            // Keep parity with Python: text_score only filters full outputs (det+rec),
            // rec-only mode returns raw recognition results.
            if !lines.is_empty() {
                output.txts = Some(lines.iter().map(|v| v.text.clone()).collect());
                output.scores = Some(lines.iter().map(|v| v.score).collect());
                output.lines = Some(lines);
            }
        }

        output.e2e_ms = Some(e2e_start.elapsed().as_secs_f32() * 1000.0);
        Ok(output)
    }

    pub fn provider_resolutions(&self) -> PipelineProviderResolutions {
        PipelineProviderResolutions {
            det: self.detector.provider_resolution(),
            cls: self.classifier.provider_resolution(),
            rec: self.recognizer.provider_resolution(),
        }
    }
}

fn init_rayon_global_pool(config: &EngineConfig) {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(threads) = resolve_rayon_threads(config) {
            builder = builder.num_threads(threads.max(1));
        }
        let _ = builder.build_global();
    });
}

fn resolve_rayon_threads(config: &EngineConfig) -> Option<usize> {
    let runtimes = [
        &config.det.runtime,
        &config.cls.runtime,
        &config.rec.runtime,
    ];
    let explicit = runtimes
        .iter()
        .filter_map(|rt| rt.rayon_threads.filter(|v| *v > 0))
        .max();
    if explicit.is_some() {
        return explicit;
    }
    if runtimes.iter().any(|rt| rt.auto_tune_threads) {
        return available_parallelism();
    }
    None
}

fn available_parallelism() -> Option<usize> {
    let physical_cores = num_cpus::get_physical().max(1);
    thread::available_parallelism()
        .ok()
        .map(|v| v.get().clamp(1, physical_cores))
}

type FullFilterOutput = (
    Vec<crate::Quad>,
    Vec<f32>,
    Vec<LineResult>,
    Option<Vec<Vec<WordBox>>>,
);

fn filter_empty_lines_boxes_and_scores(
    boxes: Vec<crate::Quad>,
    det_scores: Vec<f32>,
    lines: Vec<LineResult>,
) -> (Vec<crate::Quad>, Vec<f32>, Vec<LineResult>, Vec<usize>) {
    if boxes.len() != lines.len() || det_scores.len() != lines.len() {
        return (boxes, det_scores, lines, Vec::new());
    }
    let mut out_boxes = Vec::new();
    let mut out_scores = Vec::new();
    let mut out_lines = Vec::new();
    let mut kept_indices = Vec::new();
    for (idx, ((b, s), l)) in boxes
        .into_iter()
        .zip(det_scores.into_iter())
        .zip(lines.into_iter())
        .enumerate()
    {
        if l.text.trim().is_empty() {
            continue;
        }
        out_boxes.push(b);
        out_scores.push(s);
        out_lines.push(l);
        kept_indices.push(idx);
    }
    (out_boxes, out_scores, out_lines, kept_indices)
}

fn select_items_by_indices<T>(items: Vec<T>, indices: &[usize]) -> Vec<T> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(indices.len());
    let mut keep_iter = indices.iter().copied().peekable();
    for (idx, item) in items.into_iter().enumerate() {
        let Some(target_idx) = keep_iter.peek().copied() else {
            break;
        };
        if idx == target_idx {
            out.push(item);
            keep_iter.next();
        }
    }
    out
}

fn filter_by_text_score_for_full(
    boxes: Vec<crate::Quad>,
    det_scores: Vec<f32>,
    lines: Vec<LineResult>,
    word_boxes: Option<Vec<Vec<WordBox>>>,
    text_score: f32,
) -> FullFilterOutput {
    let mut out_boxes = Vec::new();
    let mut out_scores = Vec::new();
    let mut out_lines = Vec::new();
    let mut out_word_boxes = Vec::new();

    for (idx, line) in lines.into_iter().enumerate() {
        if idx >= boxes.len() || idx >= det_scores.len() {
            break;
        }
        if line.score < text_score {
            continue;
        }
        out_boxes.push(boxes[idx]);
        out_scores.push(det_scores[idx]);
        if let Some(word_line) = word_boxes.as_ref().and_then(|v| v.get(idx))
            && !word_line.is_empty()
        {
            out_word_boxes.push(word_line.clone());
        }
        out_lines.push(line);
    }

    let out_word_boxes = if word_boxes.is_some() {
        Some(out_word_boxes)
    } else {
        None
    };

    (out_boxes, out_scores, out_lines, out_word_boxes)
}

fn detector_cfg_from_pipeline(config: &EngineConfig) -> DetectorConfig {
    config.det.clone()
}

fn classifier_cfg_from_pipeline(config: &EngineConfig) -> ClassifierConfig {
    config.cls.clone()
}

#[derive(Debug)]
pub struct RapidOcrEngine {
    inner: RapidOcr,
}

impl RapidOcrEngine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        Ok(Self {
            inner: RapidOcr::new(config)?,
        })
    }

    pub fn run(&mut self, input: OcrInput, options: RunOptions) -> Result<OcrResult> {
        let out = self.inner.run(input, options)?;
        OcrResult::try_from(out)
    }

    pub fn provider_resolutions(&self) -> PipelineProviderResolutions {
        self.inner.provider_resolutions()
    }
}

#[allow(dead_code)]
fn _flatten_word_boxes(word_lines: &[Vec<WordBox>]) -> Vec<WordBox> {
    word_lines
        .iter()
        .flat_map(|line| line.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{
        pipeline::rapid_ocr::{filter_by_text_score_for_full, filter_empty_lines_boxes_and_scores},
        types::{LineResult, WordBox},
    };

    #[test]
    fn filter_empty_lines_keeps_boxes_scores_lines_aligned() {
        let boxes = vec![
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]],
        ];
        let det_scores = vec![0.9, 0.8];
        let lines = vec![
            LineResult {
                text: "ok".to_string(),
                score: 0.9,
                word_info: None,
            },
            LineResult {
                text: "   ".to_string(),
                score: 0.8,
                word_info: None,
            },
        ];

        let (out_boxes, out_scores, out_lines, kept_indices) =
            filter_empty_lines_boxes_and_scores(boxes, det_scores, lines);

        assert_eq!(out_boxes.len(), 1);
        assert_eq!(out_scores.len(), 1);
        assert_eq!(out_lines.len(), 1);
        assert_eq!(kept_indices, vec![0]);
        assert_eq!(out_lines[0].text, "ok");
    }

    #[test]
    fn filter_by_text_score_keeps_full_outputs_in_sync() {
        let boxes = vec![
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]],
            [[4.0, 4.0], [5.0, 4.0], [5.0, 5.0], [4.0, 5.0]],
        ];
        let det_scores = vec![0.1, 0.2, 0.3];
        let lines = vec![
            LineResult {
                text: "a".to_string(),
                score: 0.95,
                word_info: None,
            },
            LineResult {
                text: "b".to_string(),
                score: 0.50,
                word_info: None,
            },
            LineResult {
                text: "c".to_string(),
                score: 0.99,
                word_info: None,
            },
        ];
        let words = Some(vec![
            vec![WordBox {
                text: "a".to_string(),
                score: 0.95,
                bbox: boxes[0],
            }],
            vec![],
            vec![WordBox {
                text: "c".to_string(),
                score: 0.99,
                bbox: boxes[2],
            }],
        ]);

        let (out_boxes, out_scores, out_lines, out_words) =
            filter_by_text_score_for_full(boxes, det_scores, lines, words, 0.9);

        assert_eq!(out_boxes.len(), 2);
        assert_eq!(out_scores.len(), 2);
        assert_eq!(out_lines.len(), 2);
        assert_eq!(out_lines[0].text, "a");
        assert_eq!(out_lines[1].text, "c");

        let out_words = out_words.expect("word boxes should exist");
        assert_eq!(out_words.len(), 2);
        assert_eq!(out_words[0][0].text, "a");
        assert_eq!(out_words[1][0].text, "c");
    }
}
