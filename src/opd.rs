use crate::Result;
use npyz::{npz, NpyFile};
use std::{fs::File, io, path::Path};

/// Dome seeing opd map
///
/// The dome seeing is sampled on a 512x512 grid
/// Values outside the exit pupil are set to NaN
#[derive(Debug, Clone)]
pub struct OPD {
    data: Vec<f64>,
    max: f64,
    min: f64,
}

impl OPD {
    /// Creates a new OPD object
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data,
            max: f64::INFINITY,
            min: f64::NEG_INFINITY,
        }
    }
    /// Reads a CFD dome seeing OPD map
    pub fn from_npz<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = io::BufReader::new(File::open(path)?);
        let mut zip = zip::ZipArchive::new(file)?;

        let file = zip.by_name(&npz::file_name_from_array_name("opd"))?;
        let reader = NpyFile::new(file)?;
        let data = reader.into_vec::<f64>()?;

        let file = zip.by_name(&npz::file_name_from_array_name("opd max"))?;
        let reader = NpyFile::new(file)?;
        let max = reader.into_vec::<f64>()?[0];

        let file = zip.by_name(&npz::file_name_from_array_name("opd min"))?;
        let reader = NpyFile::new(file)?;
        let min = reader.into_vec::<f64>()?[0];

        Ok(Self { data, max, min })
    }
    /// Returns a reference to the opd map
    pub fn map(&self) -> &[f64] {
        self.data.as_slice()
    }
    /// Returns a mutable reference to the opd map
    pub fn mut_map(&mut self) -> &mut [f64] {
        self.data.as_mut_slice()
    }
    /// Returns the opd map x 10e`-scale`
    pub fn map_10e(&self, scale: i32) -> Vec<f64> {
        self.data.iter().map(|x| x * 10_f64.powi(-scale)).collect()
    }
    /// Returns the opd maximum
    pub fn max(&self) -> f64 {
        self.max
    }
    /// Returns the opd minimum
    pub fn min(&self) -> f64 {
        self.min
    }
    /// Return an iterator on the OPD with NaN filtered out
    pub fn no_nan_opd(&self) -> impl Iterator<Item = &f64> {
        self.data.iter().filter(|&x| !x.is_nan())
    }
    /// Return the OPD mean
    pub fn mean(&self) -> f64 {
        let opd: Vec<&f64> = self.no_nan_opd().collect();
        opd.iter().cloned().sum::<f64>() / opd.len() as f64
    }
    /// Return the OPD variance
    pub fn var(&self) -> f64 {
        let opd: Vec<&f64> = self.no_nan_opd().collect();
        let mean = self.mean();
        opd.iter().map(|&x| x - mean).map(|x| x * x).sum::<f64>() / opd.len() as f64
    }
    /// Return the OPD variance on an area specified with a mask
    pub fn masked_var(&self, mask: &[bool]) -> f64 {
        let opd: Vec<&f64> = self
            .data
            .iter()
            .zip(mask)
            .filter(|(_, &m)| m)
            .map(|(o, _)| o)
            .collect();
        let n = opd.len() as f64;
        let mean = opd.iter().cloned().sum::<f64>() / n;
        opd.iter().map(|&x| x - mean).map(|x| x * x).sum::<f64>() / n
    }
    /// Return the OPD standard deviation
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }
    /// Return the OPD standard deviation on an area specified with a mask
    pub fn masked_std(&self, mask: &[bool]) -> f64 {
        self.masked_var(mask).sqrt()
    }
    /// Return the OPD root mean square
    pub fn rms(&self) -> f64 {
        let opd: Vec<&f64> = self.no_nan_opd().collect();
        (opd.iter().map(|&x| x * x).sum::<f64>() / opd.len() as f64).sqrt()
    }
    /// Return the OPD root mean square on an area specified with a mask
    pub fn masked_rms(&self, mask: &[bool]) -> f64 {
        let opd: Vec<&f64> = self
            .data
            .iter()
            .zip(mask)
            .filter(|(_, &m)| m)
            .map(|(o, _)| o)
            .collect();
        (opd.iter().map(|&x| x * x).sum::<f64>() / opd.len() as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opd_npz() {
        let opd = OPD::from_npz("optvol_optvol_6.000000e+02.npz").unwrap();
        println!("{}/{}", opd.no_nan_opd().count(), 512 * 512);
        println!("min/max: {:.0}nm/{:.0}nm", 1e9 * opd.min(), 1e9 * opd.max());
        println!("mean: {:.0}nm", 1e9 * opd.mean());
        println!("std: {:.0}nm", 1e9 * opd.std());
    }
}
