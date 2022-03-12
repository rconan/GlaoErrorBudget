use glao_error_budget::{ASM, ASMS, OPD};
use rayon::prelude::*;
use std::time::Instant;

#[test]
fn asm_opd() {
    let opd = OPD::from_npz("optvol_optvol_6.000000e+02.npz").unwrap();
    println!("{}/{}", opd.no_nan_opd().count(), 512 * 512);
    println!("mean: {:.0}nm", 1e9 * opd.mean());
    println!("std: {:.0}nm", 1e9 * opd.std());

    println!("Assembling the ASM segments ...");
    let now = Instant::now();
    let mut asms: Vec<ASM> = ASMS::from_bins().unwrap();
    println!(" done in {}s", now.elapsed().as_secs());

    println!("Projection the opd on the ASM segments ...");
    let opd_map = opd.map();
    let now = Instant::now();
    asms.par_iter_mut().for_each(|asm| {
        asm.project(opd_map).unwrap();
    });
    println!(" done in {}s", now.elapsed().as_millis());

    let vars: Vec<_> = asms
        .iter()
        .map(|asm| asm.coefficients().iter().map(|x| x * x).sum::<f64>())
        .collect();
    let stds: Vec<_> = asms
        .iter()
        .map(|asm| 1e9 * opd.masked_rms(asm.mask()))
        .collect();
    println!("Segment WFE STD: {:.0?}nm", stds);
    println!(
        "Segment WFE RMS: {:.0?}nm",
        vars.iter().map(|x| 1e9 * x.sqrt()).collect::<Vec<_>>()
    );
    let var = asms
        .iter()
        .map(|asm| {
            asm.coefficients()
                .iter()
                .skip(1)
                .map(|x| x * x)
                .sum::<f64>()
        })
        .sum::<f64>();
    println!(
        "WFE RMS: {:.0}nm/{:.0}nm",
        opd.rms() * 1e9,
        var.sqrt() * 1e9
    );
}
