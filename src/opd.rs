use crate::Result;
use npyz::{npz, NpyFile, WriterBuilder};
use std::{fs::File, io, path::Path};

/// Dome seeing opd map
///
/// The dome seeing is sampled on a 512x512 grid
/// Values outside the exit pupil are set to NaN
pub struct OPD {
    pub data: Vec<f64>,
    pub max: f64,
    pub min: f64,
}

impl OPD {
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
    /// Returns the opd map
    pub fn map(&self) -> &[f64] {
        self.data.as_slice()
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
    /// Return the OPD standard deviation
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }
    /// Return the OPD root mean square
    pub fn rms(&self) -> f64 {
        self.no_nan_opd().map(|x| x * x).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opd() {
        let opd = OPD::from_npz("optvol_optvol_6.000000e+02.npz").unwrap();
        println!("{}/{}", opd.no_nan_opd().count(), 512 * 512);
        println!("mean: {:.0}nm", 1e9 * opd.mean());
        println!("std: {:.0}nm", 1e9 * opd.std());
    }
}
