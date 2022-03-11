use glao_error_budget::{asm, ASM, OPD};
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
    let asms = (1..=7)
        .map(|sid| {
            let mut asm: ASM = asm::from_bin(sid).unwrap();
            asm.unit_norm();
            asm
        })
        .collect::<Vec<_>>();
    println!(" done in {}s", now.elapsed().as_secs());

    println!("Projection the opd on the ASM segments ...");
    let opd_map = opd.map();
    let now = Instant::now();
    let b = asms
        .par_iter()
        .map(|asm| asm.project(opd_map).unwrap())
        .collect::<Vec<_>>();
    println!(" done in {}s", now.elapsed().as_millis());

    let var = b.iter().flatten().map(|x| x * x).sum::<f64>();
    println!(
        "WFE RMS: {:.0}nm/{:.0}nm",
        opd.rms() * 1e9,
        var.sqrt() * 1e9
    );
}
