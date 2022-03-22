use glao_error_budget::{OpdRecord, ASM, ASMS, OPD};
use std::fs::File;

fn main() -> anyhow::Result<()> {
    println!("Assembling the ASM segments ...");
    let mut asms: Vec<ASM> = ASMS::from_bins()?;

    let files = vec!["optvol_optvol_6.000000e+02.npz"];

    let records = files
        .into_iter()
        .map(|file| {
            let mut opd = OPD::from_npz(file)?;
            opd.mask_with(&asms.mask()).zero_mean();
            asms.project(&opd)?;
            Ok(OpdRecord {
                file: file.into(),
                var: opd.var(),
                segment_sum_square: asms.iter().map(|asm| opd.masked_ss(asm.mask())).collect(),
                modal_coefficients: asms
                    .iter()
                    .flat_map(|asm| asm.coefficients())
                    .cloned()
                    .collect(),
                ratios: asms.area_ratios(),
            })
        })
        .collect::<anyhow::Result<Vec<OpdRecord>>>()?;

    let record_file = File::create("domeseeing_kl.bin")?;
    bincode::serialize_into(record_file, &records)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn record() {
        let records: Vec<OpdRecord> =
            bincode::deserialize_from(File::open("domeseeing_kl.bin").unwrap()).unwrap();
        println!("OPD STD: {:.0}nm", records[0].var.sqrt() * 1e9);
        println!(
            "OPD segment RSS: {:.0?}nm",
            records[0]
                .segment_sum_square
                .iter()
                .map(|x| x.sqrt() * 1e9)
                .collect::<Vec<f64>>()
        );
        let r = records[0].ratios.iter();
        let ss = records[0]
            .segment_sum_square
            .iter()
            .zip(r.clone())
            .map(|(s, r)| s * r)
            .sum::<f64>();
        println!("OPD STD: {:.0}nm", ss.sqrt() * 1e9);
        let b_ss: Vec<_> = records[0]
            .modal_coefficients
            .chunks(500)
            .map(|b| b.iter().map(|x| x * x).sum::<f64>())
            .collect();
        println!(
            "OPD KL RSS: {:.0?}nm",
            b_ss.iter().map(|x| x.sqrt() * 1e9).collect::<Vec<f64>>()
        );
        let res_var = records[0]
            .segment_sum_square
            .iter()
            .zip(&b_ss)
            .map(|(x, y)| (x - y).abs())
            .zip(r)
            .map(|(x, r)| x * r)
            .sum::<f64>();
        println!("Residuals RSS {:.0}nm", res_var.sqrt() * 1e9);
    }
}
