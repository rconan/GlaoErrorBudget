use thiserror::Error;

pub mod asm;
pub use asm::{Segment, ASM};
mod opd;
pub use opd::OPD;

#[derive(Debug, Error)]
pub enum GlaoError {
    #[error("mode projection failed")]
    Projection,
    #[error("file no found")]
    File(#[from] std::io::Error),
    #[error("deserialization failed")]
    Bin(#[from] bincode::Error),
    #[error("zip archive failed")]
    Zip(#[from] zip::result::ZipError),
    #[error("npyz variable read failed")]
    Npyz(#[from] npyz::DTypeError),
}
pub type Result<T> = std::result::Result<T, GlaoError>;
