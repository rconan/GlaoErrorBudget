/*!
# GLAO Error Budget

ASM fitting model and dome seeing OPD data processing

 */

use thiserror::Error;

pub mod asm;
#[doc(inline)]
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

/// 7 segments ASM
pub trait ASMS {
    /// Returns the mirror shape
    ///
    /// The shape is sampled on a 512x512 regular grid
    /// Pixel outside the mirror footprint are set to NaN
    fn mirror_shape(&self, idx: Option<impl Iterator<Item = usize> + Clone>) -> Vec<f64>;
    /// Substracts the mirror shape from the opd
    ///
    /// The map is sampled on a 512x512 regular grid
    /// Pixel outside the mirror footprint are set to NaN
    fn mirror_shape_sub(&self, opd: &mut OPD, idx: Option<impl Iterator<Item = usize> + Clone>);
}
impl ASMS for Vec<ASM> {
    fn mirror_shape(&self, idx: Option<impl Iterator<Item = usize> + Clone>) -> Vec<f64> {
        let mut shape = vec![f64::NAN; 512 * 512];
        for asm in self {
            let segment_shape = asm.shape(idx.clone());
            asm.masked_replace(&mut shape, segment_shape);
        }
        shape
    }
    fn mirror_shape_sub(&self, opd: &mut OPD, idx: Option<impl Iterator<Item = usize> + Clone>) {
        let opd_map = opd.data.as_mut_slice();
        for asm in self {
            let segment_shape = asm.shape(idx.clone());
            asm.masked_sub(opd_map, segment_shape);
        }
    }
}
