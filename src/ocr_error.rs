use thiserror::Error;

#[derive(Error, Debug)]
pub enum OcrError {
    #[error("Ort error")]
    Ort(#[from] ort::Error),
    #[error("Io error")]
    Io(#[from] std::io::Error),
    #[error("Session not initialized")]
    ImageError(#[from] image::ImageError),
    #[error("Image error")]
    SessionNotInitialized,
}
