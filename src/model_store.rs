use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

use crate::error::{PaddleOcrError, Result};

pub fn default_model_store_dir() -> PathBuf {
    if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(local_app_data)
            .join("paddle-ocr-rs")
            .join("models");
    }

    PathBuf::from("models")
}

pub fn verify_existing_file(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref().to_path_buf();
    if !path.exists() {
        return Err(PaddleOcrError::FileNotFound(path));
    }
    if !path.is_file() {
        return Err(PaddleOcrError::Config(format!(
            "expected a file path, got directory: {}",
            path.display()
        )));
    }
    Ok(path)
}

pub fn ensure_downloaded(
    file_url: &str,
    expected_sha256: Option<&str>,
    save_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let save_dir = save_dir.as_ref();
    fs::create_dir_all(save_dir)?;

    let file_name = extract_file_name(file_url)?;
    let target_path = save_dir.join(file_name);

    if target_path.exists() {
        if let Some(expected) = expected_sha256 {
            let actual = sha256_file(&target_path)?;
            if actual.eq_ignore_ascii_case(expected) {
                return Ok(target_path);
            }
        } else {
            return Ok(target_path);
        }
    }

    let tmp_path = target_path.with_extension("part");
    if tmp_path.exists() {
        let _ = fs::remove_file(&tmp_path);
    }

    let client = build_http_client()?;
    let mut response = client.get(file_url).send()?;
    if !response.status().is_success() {
        return Err(PaddleOcrError::Download(format!(
            "failed to download {file_url}: HTTP {}",
            response.status()
        )));
    }

    let mut hasher = Sha256::new();
    let mut file = fs::File::create(&tmp_path)?;
    let mut buf = [0_u8; 16 * 1024];
    loop {
        let read = response.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
        file.write_all(&buf[..read])?;
    }
    file.flush()?;
    file.sync_all()?;

    if let Some(expected) = expected_sha256 {
        let actual = format!("{:x}", hasher.finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = fs::remove_file(&tmp_path);
            return Err(PaddleOcrError::HashMismatch {
                path: target_path,
                expected: expected.to_string(),
                actual,
            });
        }
    }

    if target_path.exists() {
        fs::remove_file(&target_path)?;
    }
    fs::rename(&tmp_path, &target_path)?;

    Ok(target_path)
}

fn build_http_client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(Into::into)
}

fn extract_file_name(url: &str) -> Result<String> {
    let trimmed = url.split('?').next().unwrap_or(url);
    let file_name = trimmed
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            PaddleOcrError::Download(format!("cannot extract file name from url: {url}"))
        })?;
    Ok(file_name.to_string())
}

fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let bytes = fs::read(path.as_ref())?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}
