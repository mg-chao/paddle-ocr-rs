use std::{
    env, fs,
    path::{Path, PathBuf},
};

use paddle_ocr_rs::{
    EngineConfig, LangRec, OcrInput, OcrResult, ProviderPreference, RapidOcrEngine, RunOptions,
    input::image_loader::LoadImage,
};

const CHECK_IMG_URL: &str = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.1.0/resources/test_files/ch_en_num.jpg";
const CHECK_EXPECTED: &str = "正品促销";

fn main() {
    if let Err(err) = run_main() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run_main() -> Result<(), Box<dyn std::error::Error>> {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    if raw_args.is_empty() {
        return Err(usage().into());
    }
    if matches!(raw_args[0].as_str(), "-h" | "--help") {
        println!("{}", usage());
        return Ok(());
    }

    match raw_args[0].as_str() {
        "run" => run_cmd(&raw_args[1..]),
        "config" => config_cmd(&raw_args[1..]),
        "check" => check_cmd(&raw_args[1..]),
        _ => run_cmd(&raw_args),
    }
}

#[derive(Debug, Clone)]
struct RunCli {
    img_path: String,
    config_path: Option<PathBuf>,
    provider: Option<ProviderCli>,
    device_id: Option<usize>,
    fail_if_provider_unavailable: Option<bool>,
    text_score: Option<f32>,
    lang_type: Option<LangRec>,
    use_det: Option<bool>,
    use_cls: Option<bool>,
    use_rec: Option<bool>,
    return_word_box: Option<bool>,
    return_single_char_box: Option<bool>,
    box_thresh: Option<f32>,
    unclip_ratio: Option<f32>,
    vis: bool,
    vis_word: bool,
    vis_save_dir: PathBuf,
    output_format: OutputFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderCli {
    Cpu,
    Cuda,
    DirectMl,
    Cann,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum OutputFormat {
    #[default]
    Summary,
    Json,
    Markdown,
}

fn run_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_run_cli(args)?;

    let mut cfg = if let Some(path) = &cli.config_path {
        EngineConfig::from_yaml_file(path)?
    } else {
        EngineConfig::default()
    };

    if let Some(lang) = cli.lang_type {
        cfg.rec.model.lang = lang;
    }
    if let Some(provider) = cli.provider {
        let preference = match provider {
            ProviderCli::Cpu => ProviderPreference::Cpu,
            ProviderCli::Cuda => ProviderPreference::Cuda {
                device_id: cli.device_id.unwrap_or(0),
            },
            ProviderCli::DirectMl => ProviderPreference::DirectMl {
                device_id: cli.device_id.unwrap_or(0),
            },
            ProviderCli::Cann => ProviderPreference::Cann {
                device_id: cli.device_id.unwrap_or(0),
            },
        };
        for runtime in [
            &mut cfg.det.runtime,
            &mut cfg.cls.runtime,
            &mut cfg.rec.runtime,
        ] {
            runtime.provider_preference = preference;
        }
    }
    if let Some(strict_provider) = cli.fail_if_provider_unavailable {
        for runtime in [
            &mut cfg.det.runtime,
            &mut cfg.cls.runtime,
            &mut cfg.rec.runtime,
        ] {
            runtime.fail_if_provider_unavailable = strict_provider;
        }
    }

    let mut engine = RapidOcrEngine::new(cfg)?;
    let input = parse_input(&cli.img_path);
    let run_opts = RunOptions {
        use_det: cli.use_det,
        use_cls: cli.use_cls,
        use_rec: cli.use_rec,
        return_word_box: cli.return_word_box,
        return_single_char_box: cli.return_single_char_box,
        text_score: cli.text_score,
        box_thresh: cli.box_thresh,
        unclip_ratio: cli.unclip_ratio,
    };

    let out = engine.run(input.clone(), run_opts)?;
    match cli.output_format {
        OutputFormat::Summary => print_result_summary(&out),
        OutputFormat::Json => {
            let doc = out.to_json()?;
            println!("{}", serde_json::to_string_pretty(&doc)?);
        }
        OutputFormat::Markdown => {
            println!("{}", out.to_markdown()?);
        }
    }

    if cli.vis {
        let loader = LoadImage;
        let image = loader.load(input)?;
        if let Some(vis_img) =
            out.visualize(&image, cli.vis_word || cli.return_word_box.unwrap_or(false))
        {
            fs::create_dir_all(&cli.vis_save_dir)?;
            let stem = infer_stem(&cli.img_path);
            let suffix = if cli.vis_word || cli.return_word_box.unwrap_or(false) {
                "_vis_single.png"
            } else {
                "_vis.png"
            };
            let save_path = cli.vis_save_dir.join(format!("{stem}{suffix}"));
            vis_img.save(&save_path)?;
            println!("The vis result has saved in {}", save_path.display());
        } else {
            println!("No visualization generated for current output mode.");
        }
    }

    Ok(())
}

fn config_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        return Err("usage: rapidocr config init [--output <path>]".into());
    }
    if args[0] != "init" {
        return Err("usage: rapidocr config init [--output <path>]".into());
    }

    let mut output = PathBuf::from("./default_rapidocr.yaml");
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                i += 1;
                output = PathBuf::from(
                    args.get(i)
                        .ok_or("missing value after --output")?
                        .to_string(),
                );
            }
            "-h" | "--help" => {
                println!("usage: rapidocr config init [--output <path>]");
                return Ok(());
            }
            other => return Err(format!("unknown arg `{other}`").into()),
        }
        i += 1;
    }

    let content = serde_yaml::to_string(&EngineConfig::default())?;
    fs::write(&output, content)?;
    println!("The config file has saved in {}", output.display());
    Ok(())
}

fn check_cmd(_args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = RapidOcrEngine::new(EngineConfig::default())?;
    let out = engine.run(parse_input(CHECK_IMG_URL), RunOptions::default())?;
    let text = first_text(&out).ok_or("check failed: empty OCR output")?;
    if text != CHECK_EXPECTED {
        return Err(format!(
            "The installation is incorrect! expected `{CHECK_EXPECTED}`, got `{text}`"
        )
        .into());
    }
    println!("Success! rapidocr is installed correctly!");
    Ok(())
}

fn parse_run_cli(args: &[String]) -> Result<RunCli, Box<dyn std::error::Error>> {
    let mut img_path: Option<String> = None;
    let mut config_path: Option<PathBuf> = None;
    let mut provider: Option<ProviderCli> = None;
    let mut device_id: Option<usize> = None;
    let mut fail_if_provider_unavailable = None;
    let mut text_score = None;
    let mut lang_type = None;
    let mut use_det = None;
    let mut use_cls = None;
    let mut use_rec = None;
    let mut return_word_box = None;
    let mut return_single_char_box = None;
    let mut box_thresh = None;
    let mut unclip_ratio = None;
    let mut vis = false;
    let mut vis_word = false;
    let mut vis_save_dir = PathBuf::from(".");
    let mut output_format = OutputFormat::Summary;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--img-path" | "-img" | "--img" => {
                i += 1;
                img_path = Some(
                    args.get(i)
                        .ok_or("missing value after --img-path")?
                        .to_string(),
                );
            }
            "--config" => {
                i += 1;
                config_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or("missing value after --config")?
                        .to_string(),
                ));
            }
            "--provider" => {
                i += 1;
                provider = Some(parse_provider(
                    args.get(i).ok_or("missing value after --provider")?,
                )?);
            }
            "--device-id" => {
                i += 1;
                device_id = Some(parse_usize(
                    args.get(i).ok_or("missing value after --device-id")?,
                )?);
            }
            "--fail-if-provider-unavailable" => {
                fail_if_provider_unavailable = Some(true);
            }
            "--allow-provider-fallback" => {
                fail_if_provider_unavailable = Some(false);
            }
            "--text-score" => {
                i += 1;
                text_score = Some(parse_bounded_f32(
                    args.get(i).ok_or("missing value after --text-score")?,
                    0.0,
                    1.0,
                    "--text-score",
                )?);
            }
            "--lang-type" | "--lang" => {
                i += 1;
                lang_type = Some(parse_lang(
                    args.get(i).ok_or("missing value after --lang-type")?,
                )?);
            }
            "--use-det" => {
                i += 1;
                use_det = Some(parse_bool(
                    args.get(i).ok_or("missing value after --use-det")?,
                )?);
            }
            "--use-cls" => {
                i += 1;
                use_cls = Some(parse_bool(
                    args.get(i).ok_or("missing value after --use-cls")?,
                )?);
            }
            "--use-rec" => {
                i += 1;
                use_rec = Some(parse_bool(
                    args.get(i).ok_or("missing value after --use-rec")?,
                )?);
            }
            "--return-word-box" | "--word" => {
                return_word_box = Some(true);
            }
            "--return-single-char-box" => {
                return_single_char_box = Some(true);
            }
            "--no-return-word-box" => {
                return_word_box = Some(false);
            }
            "--no-return-single-char-box" => {
                return_single_char_box = Some(false);
            }
            "--box-thresh" => {
                i += 1;
                box_thresh = Some(parse_bounded_f32(
                    args.get(i).ok_or("missing value after --box-thresh")?,
                    0.0,
                    1.0,
                    "--box-thresh",
                )?);
            }
            "--unclip-ratio" => {
                i += 1;
                unclip_ratio = Some(parse_positive_f32(
                    args.get(i).ok_or("missing value after --unclip-ratio")?,
                    "--unclip-ratio",
                )?);
            }
            "--vis" | "-vis" => {
                vis = true;
            }
            "--vis-word" => {
                vis = true;
                vis_word = true;
            }
            "--vis-save-dir" => {
                i += 1;
                vis_save_dir = PathBuf::from(
                    args.get(i)
                        .ok_or("missing value after --vis-save-dir")?
                        .to_string(),
                );
            }
            "--json" => {
                output_format = choose_output_format(output_format, OutputFormat::Json)?;
            }
            "--markdown" => {
                output_format = choose_output_format(output_format, OutputFormat::Markdown)?;
            }
            "--output-format" => {
                i += 1;
                output_format = choose_output_format(
                    output_format,
                    parse_output_format(args.get(i).ok_or("missing value after --output-format")?)?,
                )?;
            }
            "-h" | "--help" => {
                println!("{}", run_usage());
                std::process::exit(0);
            }
            other => {
                if img_path.is_none() && !other.starts_with('-') {
                    img_path = Some(other.to_string());
                } else {
                    return Err(format!("unknown arg `{other}`").into());
                }
            }
        }
        i += 1;
    }

    if device_id.is_some() && provider.is_none() {
        return Err("--device-id requires --provider".into());
    }

    Ok(RunCli {
        img_path: img_path.ok_or("missing --img-path")?,
        config_path,
        provider,
        device_id,
        fail_if_provider_unavailable,
        text_score,
        lang_type,
        use_det,
        use_cls,
        use_rec,
        return_word_box,
        return_single_char_box,
        box_thresh,
        unclip_ratio,
        vis,
        vis_word,
        vis_save_dir,
        output_format,
    })
}

fn parse_input(value: &str) -> OcrInput {
    if value.starts_with("http://") || value.starts_with("https://") {
        OcrInput::Url(value.to_string())
    } else {
        OcrInput::Path(PathBuf::from(value))
    }
}

fn parse_bool(value: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Ok(true),
        "0" | "false" | "no" => Ok(false),
        _ => Err(format!("invalid bool: `{value}`").into()),
    }
}

fn parse_bounded_f32(
    value: &str,
    min: f32,
    max: f32,
    arg_name: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    let parsed = value.parse::<f32>()?;
    if !(min..=max).contains(&parsed) {
        return Err(format!("{arg_name} must be in range [{min}, {max}], got {parsed}").into());
    }
    Ok(parsed)
}

fn parse_positive_f32(value: &str, arg_name: &str) -> Result<f32, Box<dyn std::error::Error>> {
    let parsed = value.parse::<f32>()?;
    if parsed <= 0.0 {
        return Err(format!("{arg_name} must be > 0, got {parsed}").into());
    }
    Ok(parsed)
}

fn parse_usize(value: &str) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(value.parse::<usize>()?)
}

fn parse_provider(value: &str) -> Result<ProviderCli, Box<dyn std::error::Error>> {
    Ok(match value.to_ascii_lowercase().as_str() {
        "cpu" => ProviderCli::Cpu,
        "cuda" => ProviderCli::Cuda,
        "directml" | "direct_ml" | "dml" => ProviderCli::DirectMl,
        "cann" => ProviderCli::Cann,
        _ => {
            return Err(format!(
                "unsupported --provider value `{value}` (expected one of: cpu|cuda|directml|cann)"
            )
            .into());
        }
    })
}

fn parse_output_format(value: &str) -> Result<OutputFormat, Box<dyn std::error::Error>> {
    Ok(match value.to_ascii_lowercase().as_str() {
        "summary" => OutputFormat::Summary,
        "json" => OutputFormat::Json,
        "markdown" | "md" => OutputFormat::Markdown,
        _ => {
            return Err(format!(
                "unsupported --output-format value `{value}` (expected: summary|json|markdown)"
            )
            .into());
        }
    })
}

fn choose_output_format(
    current: OutputFormat,
    next: OutputFormat,
) -> Result<OutputFormat, Box<dyn std::error::Error>> {
    if current == OutputFormat::Summary || current == next {
        return Ok(next);
    }
    Err("output format flags conflict; choose exactly one of summary/json/markdown".into())
}

fn parse_lang(value: &str) -> Result<LangRec, Box<dyn std::error::Error>> {
    Ok(match value.to_ascii_lowercase().as_str() {
        "ch" => LangRec::Ch,
        "ch_doc" => LangRec::ChDoc,
        "en" => LangRec::En,
        "arabic" => LangRec::Arabic,
        "chinese_cht" => LangRec::ChineseCht,
        "cyrillic" => LangRec::Cyrillic,
        "devanagari" => LangRec::Devanagari,
        "japan" => LangRec::Japan,
        "korean" => LangRec::Korean,
        "ka" => LangRec::Ka,
        "latin" => LangRec::Latin,
        "ta" => LangRec::Ta,
        "te" => LangRec::Te,
        "eslav" => LangRec::Eslav,
        "th" => LangRec::Th,
        "el" => LangRec::El,
        _ => return Err(format!("unsupported --lang-type value `{value}`").into()),
    })
}

fn first_text(out: &OcrResult) -> Option<&str> {
    match out {
        OcrResult::Full(v) => v.txts.first().map(String::as_str),
        OcrResult::Rec(v) => v.txts.first().map(String::as_str),
        _ => None,
    }
}

fn print_result_summary(out: &OcrResult) {
    match out {
        OcrResult::Empty => println!("No text detected."),
        OcrResult::Det(v) => {
            println!("det_boxes: {}", v.boxes.len());
            println!(
                "timing_ms(det): {:.3}",
                v.timings.det_ms.unwrap_or_default()
            );
            if let Some(e2e_ms) = v.timings.e2e_ms {
                println!("timing_ms(e2e): {:.3}", e2e_ms);
            }
        }
        OcrResult::Cls(v) => {
            println!("cls_count: {}", v.cls_res.len());
            for (idx, (label, score)) in v.cls_res.iter().enumerate() {
                println!("{idx}: {label} ({score:.5})");
            }
            println!(
                "timing_ms(cls): {:.3}",
                v.timings.cls_ms.unwrap_or_default()
            );
            if let Some(e2e_ms) = v.timings.e2e_ms {
                println!("timing_ms(e2e): {:.3}", e2e_ms);
            }
        }
        OcrResult::Rec(v) => {
            println!("line_count: {}", v.txts.len());
            for (idx, (text, score)) in v.txts.iter().zip(v.scores.iter()).enumerate() {
                println!("{idx}: '{text}' ({score:.5})");
            }
            println!(
                "timing_ms(rec): {:.3}",
                v.timings.rec_ms.unwrap_or_default()
            );
            if let Some(e2e_ms) = v.timings.e2e_ms {
                println!("timing_ms(e2e): {:.3}", e2e_ms);
            }
        }
        OcrResult::Full(v) => {
            println!("line_count: {}", v.txts.len());
            println!("det_boxes: {}", v.boxes.len());
            for (idx, (text, score)) in v.txts.iter().zip(v.scores.iter()).enumerate() {
                println!("{idx}: '{text}' ({score:.5})");
            }
            println!(
                "timing_ms(det/cls/rec/total): {:.3}/{:.3}/{:.3}/{:.3}",
                v.timings.det_ms.unwrap_or_default(),
                v.timings.cls_ms.unwrap_or_default(),
                v.timings.rec_ms.unwrap_or_default(),
                v.timings.total_ms
            );
            if let Some(e2e_ms) = v.timings.e2e_ms {
                println!("timing_ms(e2e): {:.3}", e2e_ms);
            }
        }
    }
}

fn infer_stem(img_path: &str) -> String {
    if img_path.starts_with("http://") || img_path.starts_with("https://") {
        let last = img_path.rsplit('/').next().unwrap_or("rapidocr");
        Path::new(last)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("rapidocr")
            .to_string()
    } else {
        Path::new(img_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("rapidocr")
            .to_string()
    }
}

fn usage() -> String {
    format!(
        "rapidocr commands:\n  run   {}\n  config init [--output <path>]\n  check",
        run_usage()
    )
}

fn run_usage() -> String {
    "run --img-path <path_or_url> [--config <yaml>] [--provider <cpu|cuda|directml|cann>] [--device-id <usize>] [--fail-if-provider-unavailable|--allow-provider-fallback] [--text-score <f32>] [--lang-type <lang>] [--use-det <bool>] [--use-cls <bool>] [--use-rec <bool>] [--return-word-box|--no-return-word-box] [--return-single-char-box|--no-return-single-char-box] [--box-thresh <f32>] [--unclip-ratio <f32>] [--vis] [--vis-word] [--vis-save-dir <dir>] [--output-format <summary|json|markdown>] [--json] [--markdown]".to_string()
}

#[cfg(test)]
mod tests {
    use super::{ProviderCli, parse_run_cli};

    fn args(input: &[&str]) -> Vec<String> {
        input.iter().map(|v| (*v).to_string()).collect()
    }

    #[test]
    fn parse_run_cli_provider_and_device_id() {
        let cli = parse_run_cli(&args(&[
            "--img-path",
            "test.png",
            "--provider",
            "directml",
            "--device-id",
            "2",
        ]))
        .expect("cli parse should pass");
        assert_eq!(cli.provider, Some(ProviderCli::DirectMl));
        assert_eq!(cli.device_id, Some(2));
    }

    #[test]
    fn parse_run_cli_rejects_device_id_without_provider() {
        let err = parse_run_cli(&args(&["--img-path", "test.png", "--device-id", "1"]))
            .expect_err("must reject dangling --device-id");
        assert!(err.to_string().contains("--device-id requires --provider"));
    }

    #[test]
    fn parse_run_cli_provider_strictness_toggle() {
        let strict = parse_run_cli(&args(&[
            "--img-path",
            "test.png",
            "--fail-if-provider-unavailable",
        ]))
        .expect("cli parse should pass");
        assert_eq!(strict.fail_if_provider_unavailable, Some(true));

        let fallback = parse_run_cli(&args(&[
            "--img-path",
            "test.png",
            "--allow-provider-fallback",
        ]))
        .expect("cli parse should pass");
        assert_eq!(fallback.fail_if_provider_unavailable, Some(false));
    }

    #[test]
    fn parse_run_cli_rejects_removed_config_format_flag() {
        let err = parse_run_cli(&args(&[
            "--img-path",
            "test.png",
            "--config-format",
            "rapidocr",
        ]))
        .expect_err("removed flag should be rejected");
        assert!(err.to_string().contains("unknown arg `--config-format`"));
    }
}
