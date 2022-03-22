/*!
# GLAO Error Budget

ASM fitting model and dome seeing OPD data processing

 */

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    iter::Once,
    ops::{Sub, SubAssign},
};
use thiserror::Error;

pub mod asm;
#[doc(inline)]
pub use asm::ASM;
mod opd;
pub use opd::OPD;

#[derive(Debug, Error)]
pub enum GlaoError {
    #[error("mode projection failed")]
    Projection,
    #[error("ASM from bincode failde")]
    Bin2Asm,
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

/// A single OPD data processing result
#[derive(Serialize, Deserialize)]
pub struct OpdRecord {
    /// OPD file name
    pub file: String,
    /// OPD variance
    pub var: f64,
    /// OPD segment sum square
    pub segment_sum_square: Vec<f64>,
    /// OPD Karhunen-Loeve modal coefficients
    pub modal_coefficients: Vec<f64>,
    /// segment area to exit pupil area ratios
    pub ratios: Vec<f64>,
}

/// 7 segments ASM
pub trait ASMS {
    /// Loads segment Karhunen-Loeve modes
    ///
    /// The modes are loaded from [bincode] data files in the `gerpy` directory.
    /// 500 modes are expected.
    fn from_bins() -> Result<Self>
    where
        Self: Sized;
    /// Return a mask for the ASMS
    fn mask(&self) -> Vec<bool>;
    /// Returns the mirror shape
    ///
    /// The shape is sampled on a 512x512 regular grid
    /// Pixel outside the mirror footprint are set to NaN
    fn mirror_shape(&self, idx: Option<impl Iterator<Item = usize> + Clone>) -> OPD;
    /// Substracts the mirror shape from the opd
    ///
    /// The map is sampled on a 512x512 regular grid.
    /// Pixel outside the mirror footprint are set to NaN
    fn mirror_shape_sub(&self, opd: &mut OPD, idx: Option<impl Iterator<Item = usize> + Clone>);
    /// Projects `opd` on all the modes
    fn project(&mut self, opd: &OPD) -> Result<&mut Self>;
    /// Segment area to exit pupil area ratios
    fn area_ratios(&self) -> Vec<f64>;
}
impl ASMS for Vec<ASM> {
    fn from_bins() -> Result<Self> {
        (1..=7)
            .into_par_iter()
            .map(|sid| ASM::from_bin(sid))
            .collect()
    }
    fn mask(&self) -> Vec<bool> {
        self.iter().fold(vec![false; 512 * 512], |mut a, asm| {
            a.iter_mut()
                .zip(asm.mask())
                .for_each(|(a, m)| *a = *a || *m);
            a
        })
    }
    fn mirror_shape(&self, idx: Option<impl Iterator<Item = usize> + Clone>) -> OPD {
        let mut shape = vec![f64::NAN; 512 * 512];
        for asm in self {
            let segment_shape = asm.shape(idx.clone());
            asm.masked_replace(&mut shape, segment_shape);
        }
        OPD::new(shape)
    }
    fn mirror_shape_sub(&self, opd: &mut OPD, idx: Option<impl Iterator<Item = usize> + Clone>) {
        let opd_map = opd.mut_map();
        self.mask()
            .into_iter()
            .zip(opd_map.iter_mut())
            .filter(|(m, _)| !*m)
            .for_each(|(_, o)| *o = f64::NAN);
        for asm in self {
            let segment_shape = asm.shape(idx.clone());
            asm.masked_sub(opd_map, segment_shape);
        }
    }
    fn project(&mut self, opd: &OPD) -> Result<&mut Self> {
        let opd_map = opd.map();
        self.par_iter_mut()
            .map(|asm| asm.project(opd_map))
            .collect::<Result<Vec<_>>>()?;
        Ok(self)
    }
    fn area_ratios(&self) -> Vec<f64> {
        let n_points: Vec<_> = self.iter().map(|asm| asm.n_point()).collect();
        let nn_points: usize = n_points.iter().sum();
        n_points
            .into_iter()
            .map(|x| x as f64 / nn_points as f64)
            .collect()
    }
}

impl<T> SubAssign<&T> for OPD
where
    T: ASMS,
{
    fn sub_assign(&mut self, rhs: &T) {
        rhs.mirror_shape_sub(self, Option::<Once<usize>>::None)
    }
}

impl<T> Sub<&T> for &OPD
where
    T: ASMS,
{
    type Output = OPD;

    fn sub(self, rhs: &T) -> Self::Output {
        let mut opd = self.clone();
        opd -= rhs;
        opd
    }
}
