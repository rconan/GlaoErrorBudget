use crate::{GlaoError, Result};
use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::{fs::File, path::Path};

/// Karhunen-Loeve modal basis
#[derive(Serialize, Deserialize, Debug)]
pub struct KarhunenLoeve {
    /// Segment modes
    pub modes: Vec<f64>,
    /// Number of modes
    pub n_mode: usize,
    /// Pupil mask for the segment
    pub mask: Vec<bool>,
}
impl KarhunenLoeve {
    /// Loads segment Karhunen-Loeve modes
    ///
    /// The modes are loaded from [bincode] data files in the `gerpy` directory.
    /// 500 modes are expected.
    /// The data files are generated with the `gerpy/export.py` script from `segKLmat.npz`.
    /// The python data transfer interface is created with the binary `gerpy`.
    pub fn from_bin(sid: usize) -> Result<Self> {
        let path = Path::new("gerpy");
        let filename = path.join(format!("M2S{sid}")).with_extension("bin");
        println!("Loading {filename:?}");
        let file = File::open(filename)?;
        Ok(bincode::deserialize_from(file)?)
    }
}
/// A segment
///
/// A segment consists of `n_mode` Karhunen-Loeve modes concatenated in the `modes` vector.
/// Each mode is defined on a particular segment and the location of the modes
/// are set with the 512x512 exit pupil `mask`
#[derive(Debug)]
pub struct Segment {
    /// Segment modes
    pub modes: Vec<f64>,
    /// Number of modes
    pub n_mode: usize,
    /// Pupil mask for the segment
    pub mask: Vec<bool>,
    /// Modal coefficients
    pub coefficients: Vec<f64>,
    /// Segment modes pseudo-inverse
    pub modes_pinv: na::DMatrix<f64>,
}
impl From<KarhunenLoeve> for Segment {
    fn from(kl: KarhunenLoeve) -> Self {
        let modes = na::DMatrix::from_column_slice(
            kl.modes.len() / kl.n_mode,
            kl.n_mode,
            kl.modes.as_slice(),
        );
        let modes_pinv = modes
            .pseudo_inverse(0f64)
            .map_err(|e| GlaoError::PseudoInverse(e.into()))
            .expect("pseudo-inverse failed");
        Self {
            modes: kl.modes,
            n_mode: kl.n_mode,
            mask: kl.mask,
            coefficients: Vec::new(),
            modes_pinv,
        }
    }
}
impl Segment {
    /// Creates a new segment
    ///
    /// The modes `m` are normalized  such as `|m|=1`
    pub fn new(n_mode: usize, modes: Vec<f64>, mask: Vec<bool>) -> Self {
        let mat_modes =
            na::DMatrix::from_column_slice(modes.len() / n_mode, n_mode, modes.as_slice());
        let modes_pinv = mat_modes
            .pseudo_inverse(0f64)
            .map_err(|e| GlaoError::PseudoInverse(e.into()))
            .expect("pseudo-inverse failed");
        Self {
            modes,
            n_mode,
            mask,
            coefficients: Vec::new(),
            modes_pinv,
        }
    }
    /// Returns the number of points within the segment
    pub fn n_point(&self) -> usize {
        self.modes.len() / self.n_mode
    }
    /// Returns the number of points within the mask
    pub fn n_in_mask(&self) -> usize {
        self.mask.iter().filter_map(|m| m.then(|| 1)).sum()
    }
    /// Applies the mask
    pub fn masked(&self, data: &[f64]) -> Vec<f64> {
        self.mask
            .iter()
            .zip(data)
            .filter(|(&m, _)| m)
            .map(|(_, o)| *o)
            .collect()
    }
    /// Replaces the `old_data` within the mask with the `new_data`
    ///
    /// The old data has the same size than the mask array.
    /// The new data is the size of the masked area
    pub fn masked_replace(&self, old_data: &mut [f64], new_data: Vec<f64>) {
        self.mask
            .iter()
            .zip(old_data)
            .filter(|(&m, _)| m)
            .map(|(_, o)| o)
            .zip(new_data.into_iter())
            .for_each(|(old, new)| {
                *old = new;
            });
    }
    /// Substracts `new_data` from `old_data` within the mask
    ///
    /// The old data has the same size than the mask array.
    /// The new data is the size of the masked area
    pub fn masked_sub(&self, old_data: &mut [f64], new_data: Vec<f64>) {
        self.mask
            .iter()
            .zip(old_data)
            .filter(|(&m, _)| m)
            .map(|(_, o)| o)
            .zip(new_data.into_iter())
            .for_each(|(old, new)| {
                *old -= new;
            });
    }
    /// Projects `opd` on all the modes
    pub fn project(&mut self, opd: &[f64]) -> Result<&mut Self> {
        let n = self.n_point();
        let m: usize = 512 * 512;
        self.coefficients = match opd.len() {
            l if l == m => {
                let masked_opd: Vec<_> = self.masked(opd);
                Ok(self
                    .modes
                    .chunks(n)
                    .map(|mode| {
                        let norm = (mode.iter().map(|x| x * x).sum::<f64>() * n as f64)
                            .sqrt()
                            .recip();
                        norm * mode
                            .iter()
                            .zip(masked_opd.iter())
                            .fold(0f64, |a, (&x, y)| a + x * y)
                    })
                    .collect())
            }
            l if l == n => Ok(self
                .modes
                .chunks(n)
                .map(|mode| {
                    let norm = (mode.iter().map(|x| x * x).sum::<f64>() * n as f64)
                        .sqrt()
                        .recip();
                    norm * mode.iter().zip(opd).fold(0f64, |a, (&x, &y)| a + x * y)
                })
                .collect()),
            _ => Err(GlaoError::Projection),
        }?;
        Ok(self)
    }
    pub fn least_square(&mut self, opd: &[f64]) -> Result<&mut Self> {
        let n = self.n_point();
        let m: usize = 512 * 512;
        let masked_opd = match opd.len() {
            l if l == m => Ok(na::DVector::from_column_slice(self.masked(opd).as_slice())),
            l if l == n => Ok(na::DVector::from_column_slice(opd)),
            _ => Err(GlaoError::Projection),
        }?;
        self.coefficients = (&self.modes_pinv * masked_opd).as_slice().to_vec();
        Ok(self)
    }
    pub fn project_out(&self, opd: &[f64]) -> Result<Vec<f64>> {
        let n = self.n_point();
        let m: usize = 512 * 512;
        match opd.len() {
            l if l == m => {
                let masked_opd: Vec<_> = self.masked(opd);
                Ok(self
                    .modes
                    .chunks(n)
                    .map(|mode| {
                        let norm = (mode.iter().map(|x| x * x).sum::<f64>() * n as f64)
                            .sqrt()
                            .recip();
                        norm * mode
                            .iter()
                            .zip(masked_opd.iter())
                            .fold(0f64, |a, (&x, y)| a + x * y)
                    })
                    .collect())
            }
            l if l == n => Ok(self
                .modes
                .chunks(n)
                .map(|mode| {
                    let norm = (mode.iter().map(|x| x * x).sum::<f64>() * n as f64)
                        .sqrt()
                        .recip();
                    norm * mode.iter().zip(opd).fold(0f64, |a, (&x, &y)| a + x * y)
                })
                .collect()),
            _ => Err(GlaoError::Projection),
        }
    }
    pub fn least_square_out(&self, opd: &[f64]) -> Result<Vec<f64>> {
        let n = self.n_point();
        let m: usize = 512 * 512;
        let masked_opd = match opd.len() {
            l if l == m => Ok(na::DVector::from_column_slice(self.masked(opd).as_slice())),
            l if l == n => Ok(na::DVector::from_column_slice(opd)),
            _ => Err(GlaoError::Projection),
        }?;
        let b = &self.modes_pinv * masked_opd;
        Ok(b.as_slice().to_vec())
    }
    /// Computes the shape of the mirror segment
    ///
    /// Uses either all the modes or a specified set in an [Iterator]
    pub fn shape(&self, idx: Option<impl Iterator<Item = usize> + Clone>) -> Vec<f64> {
        let n = self.n_point();
        if let Some(idx) = idx {
            let modes = idx
                .clone()
                .map(|idx| self.modes.chunks(n).nth(idx).unwrap());
            let coefficients = idx.map(|idx| self.coefficients[idx]);
            modes
                .zip(coefficients)
                .fold(vec![0f64; n], |mut w, (m, c)| {
                    w.iter_mut().zip(m).for_each(|(w, &m)| *w += m * c);
                    w
                })
        } else {
            self.modes
                .chunks(n)
                .zip(&self.coefficients)
                .fold(vec![0f64; n], |mut w, (m, c)| {
                    w.iter_mut().zip(m).for_each(|(w, &m)| *w += m * c);
                    w
                })
        }
    }
}

/// A single ASM
#[derive(Debug)]
pub enum ASM {
    S1(Segment),
    S2(Segment),
    S3(Segment),
    S4(Segment),
    S5(Segment),
    S6(Segment),
    S7(Segment),
}
impl ASM {
    pub fn tag(&self) -> String {
        use ASM::*;
        match self {
            S1(_) => "M2S1",
            S2(_) => "M2S2",
            S3(_) => "M2S3",
            S4(_) => "M2S4",
            S5(_) => "M2S5",
            S6(_) => "M2S6",
            S7(_) => "M2S7",
        }
        .to_string()
    }
    pub fn from_bin(sid: usize) -> Result<Self> {
        let kl = KarhunenLoeve::from_bin(sid)?;
        match sid {
            id if id == 1 => Ok(ASM::S1(kl.into())),
            id if id == 2 => Ok(ASM::S2(kl.into())),
            id if id == 3 => Ok(ASM::S3(kl.into())),
            id if id == 4 => Ok(ASM::S4(kl.into())),
            id if id == 5 => Ok(ASM::S5(kl.into())),
            id if id == 6 => Ok(ASM::S6(kl.into())),
            id if id == 7 => Ok(ASM::S7(kl.into())),
            _ => Err(GlaoError::Bin2Asm),
        }
    }
    /// Projects `opd` on all the modes
    pub fn project(&mut self, opd: &[f64]) -> Result<&mut Self> {
        use ASM::*;
        match self {
            S1(segment) => segment.project(opd),
            S2(segment) => segment.project(opd),
            S3(segment) => segment.project(opd),
            S4(segment) => segment.project(opd),
            S5(segment) => segment.project(opd),
            S6(segment) => segment.project(opd),
            S7(segment) => segment.project(opd),
        }?;
        Ok(self)
    }
    pub fn least_square(&mut self, opd: &[f64]) -> Result<&mut Self> {
        use ASM::*;
        match self {
            S1(segment) => segment.least_square(opd),
            S2(segment) => segment.least_square(opd),
            S3(segment) => segment.least_square(opd),
            S4(segment) => segment.least_square(opd),
            S5(segment) => segment.least_square(opd),
            S6(segment) => segment.least_square(opd),
            S7(segment) => segment.least_square(opd),
        }?;
        Ok(self)
    }
    pub fn project_out(&self, opd: &[f64]) -> Result<Vec<f64>> {
        use ASM::*;
        match self {
            S1(segment) => segment.project_out(opd),
            S2(segment) => segment.project_out(opd),
            S3(segment) => segment.project_out(opd),
            S4(segment) => segment.project_out(opd),
            S5(segment) => segment.project_out(opd),
            S6(segment) => segment.project_out(opd),
            S7(segment) => segment.project_out(opd),
        }
    }
    pub fn least_square_out(&self, opd: &[f64]) -> Result<Vec<f64>> {
        use ASM::*;
        match self {
            S1(segment) => segment.least_square_out(opd),
            S2(segment) => segment.least_square_out(opd),
            S3(segment) => segment.least_square_out(opd),
            S4(segment) => segment.least_square_out(opd),
            S5(segment) => segment.least_square_out(opd),
            S6(segment) => segment.least_square_out(opd),
            S7(segment) => segment.least_square_out(opd),
        }
    }
    /// Returns the number of points within the segment
    pub fn n_point(&self) -> usize {
        use ASM::*;
        match self {
            S1(segment) => segment.n_point(),
            S2(segment) => segment.n_point(),
            S3(segment) => segment.n_point(),
            S4(segment) => segment.n_point(),
            S5(segment) => segment.n_point(),
            S6(segment) => segment.n_point(),
            S7(segment) => segment.n_point(),
        }
    }
    /// Returns the number of points within the mask
    pub fn n_in_mask(&self) -> usize {
        use ASM::*;
        match self {
            S1(segment) => segment.n_in_mask(),
            S2(segment) => segment.n_in_mask(),
            S3(segment) => segment.n_in_mask(),
            S4(segment) => segment.n_in_mask(),
            S5(segment) => segment.n_in_mask(),
            S6(segment) => segment.n_in_mask(),
            S7(segment) => segment.n_in_mask(),
        }
    }
    /// Returns the segment shape
    pub fn shape(&self, idx: Option<impl Iterator<Item = usize> + Clone>) -> Vec<f64> {
        use ASM::*;
        match self {
            S1(segment) => segment.shape(idx),
            S2(segment) => segment.shape(idx),
            S3(segment) => segment.shape(idx),
            S4(segment) => segment.shape(idx),
            S5(segment) => segment.shape(idx),
            S6(segment) => segment.shape(idx),
            S7(segment) => segment.shape(idx),
        }
    }
    /// Returns the segment modal coefficients
    pub fn coefficients(&self) -> &[f64] {
        use ASM::*;
        match self {
            S1(segment) => segment.coefficients.as_slice(),
            S2(segment) => segment.coefficients.as_slice(),
            S3(segment) => segment.coefficients.as_slice(),
            S4(segment) => segment.coefficients.as_slice(),
            S5(segment) => segment.coefficients.as_slice(),
            S6(segment) => segment.coefficients.as_slice(),
            S7(segment) => segment.coefficients.as_slice(),
        }
    }
    /// Returns the segment modes
    pub fn modes(&self) -> &[f64] {
        use ASM::*;
        match self {
            S1(segment) => segment.modes.as_slice(),
            S2(segment) => segment.modes.as_slice(),
            S3(segment) => segment.modes.as_slice(),
            S4(segment) => segment.modes.as_slice(),
            S5(segment) => segment.modes.as_slice(),
            S6(segment) => segment.modes.as_slice(),
            S7(segment) => segment.modes.as_slice(),
        }
    }
    /// Returns the segment mask
    pub fn mask(&self) -> &[bool] {
        use ASM::*;
        match self {
            S1(segment) => segment.mask.as_slice(),
            S2(segment) => segment.mask.as_slice(),
            S3(segment) => segment.mask.as_slice(),
            S4(segment) => segment.mask.as_slice(),
            S5(segment) => segment.mask.as_slice(),
            S6(segment) => segment.mask.as_slice(),
            S7(segment) => segment.mask.as_slice(),
        }
    }
    /// Replace the `old_data` with the mask with the `new_data`
    ///
    /// The old data has the same size than the mask array.
    /// The new data is the size of the masked array.
    pub fn masked_replace(&self, old_data: &mut [f64], new_data: Vec<f64>) {
        use ASM::*;
        match self {
            S1(segment) => segment.masked_replace(old_data, new_data),
            S2(segment) => segment.masked_replace(old_data, new_data),
            S3(segment) => segment.masked_replace(old_data, new_data),
            S4(segment) => segment.masked_replace(old_data, new_data),
            S5(segment) => segment.masked_replace(old_data, new_data),
            S6(segment) => segment.masked_replace(old_data, new_data),
            S7(segment) => segment.masked_replace(old_data, new_data),
        }
    }
    /// Substracts `new_data` from `old_data` within the mask
    ///
    /// The old data has the same size than the mask array.
    /// The new data is the size of the masked area
    pub fn masked_sub(&self, old_data: &mut [f64], new_data: Vec<f64>) {
        use ASM::*;
        match self {
            S1(segment) => segment.masked_sub(old_data, new_data),
            S2(segment) => segment.masked_sub(old_data, new_data),
            S3(segment) => segment.masked_sub(old_data, new_data),
            S4(segment) => segment.masked_sub(old_data, new_data),
            S5(segment) => segment.masked_sub(old_data, new_data),
            S6(segment) => segment.masked_sub(old_data, new_data),
            S7(segment) => segment.masked_sub(old_data, new_data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_from_bin() {
        let kl = KarhunenLoeve::from_bin(1).unwrap();
        assert_eq!(kl.n_mode, 500);
        assert_eq!(kl.modes.len(), 500 * 21382);
        assert_eq!(kl.mask.len(), 512 * 512);
    }

    /*
        #[test]
        fn py2rs() {
            let file = File::open("pytest.bin").unwrap();
            let asm: ASM = bincode::deserialize_from(file).unwrap();
            if let ASM::S1(Segment {
                modes,
                n_mode,
                mask,
                ..
            }) = asm
            {
                assert_eq!(n_mode, 500);
                assert_eq!(modes.len(), 500 * 21382);
                assert_eq!(mask.len(), 512 * 512);
                println!("{:?}", &modes[..10]);
            }
        }
    */
    #[test]
    fn s1_from_bin() {
        let asm = ASM::from_bin(1).unwrap();
        if let ASM::S1(Segment {
            modes,
            n_mode,
            mask,
            ..
        }) = asm
        {
            assert_eq!(n_mode, 500);
            assert_eq!(modes.len(), 500 * 21382);
            assert_eq!(mask.len(), 512 * 512);
        }
    }
    #[test]
    fn s7_from_bin() {
        let asm = ASM::from_bin(7).unwrap();
        if let ASM::S1(Segment {
            modes,
            n_mode,
            mask,
            ..
        }) = asm
        {
            assert_eq!(n_mode, 500);
            assert_eq!(modes.len(), 500 * 15686);
            assert_eq!(mask.len(), 512 * 512);
        }
    }

    #[test]
    fn project() {
        let mut asm = ASM::from_bin(1).unwrap();
        let modes = asm.modes().chunks(asm.n_point()).nth(3).unwrap().to_vec();
        asm.project(&modes).unwrap();
        println!("b: {:?}", &asm.coefficients()[..5]);
    }

    #[test]
    fn project_out() {
        let mut asm = ASM::from_bin(1).unwrap();
        let modes = asm.modes().chunks(asm.n_point()).nth(3).unwrap().to_vec();
        let b = asm.project_out(&modes).unwrap();
        println!("b: {:?}", &b[..5]);
    }

    #[test]
    fn least_square_out() {
        let mut asm = ASM::from_bin(1).unwrap();
        let modes = asm.modes().chunks(asm.n_point()).nth(3).unwrap().to_vec();
        let b = asm.least_square_out(&modes).unwrap();
        println!("b: {:?}", &b[..5]);
    }
}
