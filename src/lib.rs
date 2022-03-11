use std::{fs::File, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GlaoError {
    #[error("mode projection failed")]
    Projection,
    #[error("file no found")]
    File(#[from] std::io::Error),
    #[error("deserialization failed")]
    Bin(#[from] bincode::Error),
}
pub type Result<T> = std::result::Result<T, GlaoError>;

/// A segment
///
/// A segment consists of `n_mode` Karhunen-Loeve modes concatenated in the `modes` vector.
/// Each mode is defined on a particular segment and the location of the modes
/// are set with the 512x512 exit pupil `mask`
#[derive(Serialize, Deserialize, Debug)]
pub struct Segment {
    /// Segment modes
    pub modes: Vec<f64>,
    /// Number of modes
    pub n_mode: usize,
    /// Pupil mask for the segment
    pub mask: Vec<bool>,
}
impl Segment {
    /// Creates a new segment
    ///
    /// The modes `m` are normalized  such as `|m|=1`
    pub fn new(n_mode: usize, modes: Vec<f64>, mask: Vec<bool>) -> Self {
        Self {
            modes,
            n_mode,
            mask,
        }
    }
    /// Normalizes the modes
    ///
    /// The modes `m` are normalized  such as `|m|=1`
    pub fn unit_norm(&mut self) {
        let n = self.n_point();
        self.modes.chunks_mut(n).for_each(|mode| {
            let rnum = mode.iter().map(|x| x * x).sum::<f64>().sqrt().recip();
            mode.iter_mut().for_each(|mode| *mode *= rnum);
        });
    }
    /// Returns the number of points within the segment
    pub fn n_point(&self) -> usize {
        self.modes.len() / self.n_mode
    }
    /// Projects `opd` on all the modes
    pub fn project(&self, opd: &[f64]) -> Result<Vec<f64>> {
        let n = self.n_point();
        let m: usize = 512 * 512;
        match opd.len() {
            l if l == m => {
                let masked_opd: Vec<_> = self
                    .mask
                    .iter()
                    .zip(opd)
                    .filter(|(&m, _)| m)
                    .map(|(_, o)| *o)
                    .collect();
                Ok(self
                    .modes
                    .chunks(n)
                    .map(|m| {
                        m.iter()
                            .zip(masked_opd.iter())
                            .fold(0f64, |a, (&x, y)| a + x * y)
                    })
                    .collect())
            }
            l if l == n => Ok(self
                .modes
                .chunks(n)
                .map(|m| m.iter().zip(opd).fold(0f64, |a, (&x, &y)| a + x * y))
                .collect()),
            _ => Err(GlaoError::Projection),
        }
    }
}

/// ASM 7 segments
#[derive(Serialize, Deserialize, Debug)]
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
    /// Projects `opd` on all the modes
    pub fn project(&self, opd: &[f64]) -> Result<Vec<f64>> {
        use ASM::*;
        match self {
            S1(segment) => segment.project(opd),
            S2(segment) => segment.project(opd),
            S3(segment) => segment.project(opd),
            S4(segment) => segment.project(opd),
            S5(segment) => segment.project(opd),
            S6(segment) => segment.project(opd),
            S7(segment) => segment.project(opd),
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
    /// Normalizes the modes
    ///
    /// The modes `m` are normalized  such as `|m|=1`
    pub fn unit_norm(&mut self) -> &mut Self {
        use ASM::*;
        match self {
            S1(segment) => segment.unit_norm(),
            S2(segment) => segment.unit_norm(),
            S3(segment) => segment.unit_norm(),
            S4(segment) => segment.unit_norm(),
            S5(segment) => segment.unit_norm(),
            S6(segment) => segment.unit_norm(),
            S7(segment) => segment.unit_norm(),
        }
        self
    }
}

pub fn from_bin(sid: usize) -> Result<ASM> {
    let path = Path::new("gerpy");
    let file = File::open(path.join(format!("M2S{sid}")).with_extension("bin"))?;
    Ok(bincode::deserialize_from(file)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    #[test]
    fn asm2bin() {
        let segment = Segment::new(3, vec![1f64, 2., 3.], vec![true; 6]);
        let asm = ASM::S1(segment);
        let file = File::create("test.bin").unwrap();
        bincode::serialize_into(file, &asm).unwrap();
    }

    #[test]
    fn py2rs() {
        let file = File::open("pytest.bin").unwrap();
        let asm: ASM = bincode::deserialize_from(file).unwrap();
        if let ASM::S1(Segment {
            modes,
            n_mode,
            mask,
        }) = asm
        {
            assert_eq!(n_mode, 500);
            assert_eq!(modes.len(), 500 * 21382);
            assert_eq!(mask.len(), 512 * 512);
            println!("{:?}", &modes[..10]);
        }
    }

    #[test]
    fn s1_from_bin() {
        let asm: ASM = from_bin(1).unwrap();
        if let ASM::S1(Segment {
            modes,
            n_mode,
            mask,
        }) = asm
        {
            assert_eq!(n_mode, 500);
            assert_eq!(modes.len(), 500 * 21382);
            assert_eq!(mask.len(), 512 * 512);
        }
    }
    #[test]
    fn s7_from_bin() {
        let asm: ASM = from_bin(7).unwrap();
        if let ASM::S1(Segment {
            modes,
            n_mode,
            mask,
        }) = asm
        {
            assert_eq!(n_mode, 500);
            assert_eq!(modes.len(), 500 * 15686);
            assert_eq!(mask.len(), 512 * 512);
        }
    }

    #[test]
    fn project() {
        let mut asm: ASM = from_bin(1).unwrap();
        asm.unit_norm();
        if let ASM::S1(Segment { modes, .. }) = &asm {
            let b = asm
                .project(modes.chunks(asm.n_point()).nth(3).unwrap())
                .unwrap();
            println!("b: {:?}", &b[..5]);
        }
    }
}
