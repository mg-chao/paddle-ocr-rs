use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use paddle_ocr_rs::{EngineConfig, OcrInput, OcrResult, RapidOcrEngine, RunOptions, StageTimings};
use serde_json::json;

fn main() {
    if let Err(err) = run_main() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run_main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli()?;
    let images = collect_images(&cli.images_dir)?;
    if images.is_empty() {
        return Err(format!("no images found under {}", cli.images_dir.display()).into());
    }

    let cfg = if let Some(config_path) = &cli.config_path {
        EngineConfig::from_yaml_file(config_path)?
    } else {
        EngineConfig::default()
    };

    let init_start = Instant::now();
    let mut engine = RapidOcrEngine::new(cfg)?;
    let init_ms = ms(init_start);

    let mut wall_ms = Vec::new();
    let mut det_ms = Vec::new();
    let mut cls_ms = Vec::new();
    let mut rec_ms = Vec::new();
    let mut e2e_ms = Vec::new();
    let mut det_pre_ms = Vec::new();
    let mut det_infer_ms = Vec::new();
    let mut det_post_ms = Vec::new();

    for _round in 0..cli.rounds {
        for image in &images {
            let run_start = Instant::now();
            let out = engine.run(OcrInput::Path(image.clone()), RunOptions::default())?;
            wall_ms.push(ms(run_start));

            if let Some(t) = timings_from_result(&out) {
                if let Some(v) = t.det_ms {
                    det_ms.push(v as f64);
                }
                if let Some(v) = t.cls_ms {
                    cls_ms.push(v as f64);
                }
                if let Some(v) = t.rec_ms {
                    rec_ms.push(v as f64);
                }
                if let Some(v) = t.e2e_ms {
                    e2e_ms.push(v as f64);
                }
                if cli.profile_det_breakdown {
                    if let Some(v) = t.det_pre_ms {
                        det_pre_ms.push(v as f64);
                    }
                    if let Some(v) = t.det_infer_ms {
                        det_infer_ms.push(v as f64);
                    }
                    if let Some(v) = t.det_post_ms {
                        det_post_ms.push(v as f64);
                    }
                }
            }
        }
    }

    let det_breakdown = if cli.profile_det_breakdown {
        Some(json!({
            "pre_ms": stats(&det_pre_ms),
            "infer_ms": stats(&det_infer_ms),
            "post_ms": stats(&det_post_ms),
        }))
    } else {
        None
    };

    let report = json!({
        "meta": {
            "images_dir": cli.images_dir,
            "image_count": images.len(),
            "rounds": cli.rounds,
            "init_ms": init_ms,
            "profile_det_breakdown": cli.profile_det_breakdown,
        },
        "stats": {
            "wall_ms": stats(&wall_ms),
            "det_ms": stats(&det_ms),
            "cls_ms": stats(&cls_ms),
            "rec_ms": stats(&rec_ms),
            "e2e_ms": stats(&e2e_ms),
            "det_breakdown_ms": det_breakdown,
        }
    });

    let text = serde_json::to_string_pretty(&report)?;
    if let Some(path) = cli.output_path {
        fs::write(&path, &text)?;
        println!("saved report to {}", path.display());
    }
    println!("{text}");
    Ok(())
}

#[derive(Debug, Clone)]
struct Cli {
    config_path: Option<PathBuf>,
    images_dir: PathBuf,
    rounds: usize,
    output_path: Option<PathBuf>,
    profile_det_breakdown: bool,
}

fn parse_cli() -> Result<Cli, Box<dyn std::error::Error>> {
    let mut config_path = None;
    let mut images_dir = PathBuf::from("test/test_files");
    let mut rounds = 1usize;
    let mut output_path = None;
    let mut profile_det_breakdown = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                config_path = Some(PathBuf::from(
                    args.next().ok_or("missing value for --config")?,
                ));
            }
            "--images-dir" => {
                images_dir = PathBuf::from(args.next().ok_or("missing value for --images-dir")?);
            }
            "--rounds" => {
                rounds = args
                    .next()
                    .ok_or("missing value for --rounds")?
                    .parse::<usize>()?;
                if rounds == 0 {
                    return Err("rounds must be greater than zero".into());
                }
            }
            "--output" => {
                output_path = Some(PathBuf::from(
                    args.next().ok_or("missing value for --output")?,
                ));
            }
            "--profile-det-breakdown" => {
                profile_det_breakdown = true;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown arg `{arg}`").into()),
        }
    }

    Ok(Cli {
        config_path,
        images_dir,
        rounds,
        output_path,
        profile_det_breakdown,
    })
}

fn print_usage() {
    println!(
        "usage: bench_warm_e2e [--config <yaml>] [--images-dir <dir>] [--rounds <n>] [--output <json>] [--profile-det-breakdown]"
    );
}

fn collect_images(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|v| v.to_str()) else {
            continue;
        };
        if is_image_ext(ext) {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn is_image_ext(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "jpg" | "jpeg" | "png" | "bmp" | "webp" | "tif" | "tiff"
    )
}

fn timings_from_result(result: &OcrResult) -> Option<&StageTimings> {
    match result {
        OcrResult::Det(v) => Some(&v.timings),
        OcrResult::Cls(v) => Some(&v.timings),
        OcrResult::Rec(v) => Some(&v.timings),
        OcrResult::Full(v) => Some(&v.timings),
        OcrResult::Empty => None,
    }
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn stats(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return json!({
            "count": 0
        });
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let count = sorted.len();
    let sum = sorted.iter().copied().sum::<f64>();
    let avg = sum / count as f64;
    let min = sorted[0];
    let max = sorted[count - 1];
    let p50 = percentile(&sorted, 0.5);
    let p90 = percentile(&sorted, 0.9);

    json!({
        "count": count,
        "avg": avg,
        "p50": p50,
        "p90": p90,
        "min": min,
        "max": max,
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}
