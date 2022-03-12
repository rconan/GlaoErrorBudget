use glao_error_budget::{asm, Result, ASM, ASMS, OPD};
use rayon::prelude::*;
use std::{iter::Once, time::Instant};

fn main() -> anyhow::Result<()> {
    // loading the OPD
    let opd = OPD::from_npz("optvol_optvol_6.000000e+02.npz")?;
    println!("{}/{}", opd.no_nan_opd().count(), 512 * 512);
    println!("OPD mean: {:.0}nm", 1e9 * opd.mean());
    println!("OPD std: {:.0}nm", 1e9 * opd.std());

    // collecting the 7 modal basis
    println!("Assembling the ASM segments ...");
    let now = Instant::now();
    let mut asms = (1..=7)
        .map(|sid| {
            let mut asm: ASM = asm::from_bin(sid).unwrap();
            asm.unit_norm();
            asm
        })
        .collect::<Vec<_>>();
    println!(" done in {}s", now.elapsed().as_secs());

    // OPD projection
    println!("Projection the opd on the ASM segments ...");
    let opd_map = opd.map();
    let now = Instant::now();
    asms.par_iter_mut().for_each(|asm| {
        asm.project(opd_map).unwrap();
    });
    println!(" done in {}s", now.elapsed().as_millis());

    // Segment WFE
    // - from the modal coefficients
    let vars: Vec<_> = asms
        .iter()
        .map(|asm| asm.coefficients().iter().map(|x| x * x).sum::<f64>())
        .collect();
    // - from the wavefront
    let stds: Vec<_> = asms
        .iter()
        .map(|asm| 1e9 * opd.masked_rms(asm.mask()))
        .collect();
    println!("Segment WFE STD: {:.0?}nm", stds);
    println!(
        "Segment WFE RMS: {:.0?}nm",
        vars.iter().map(|x| 1e9 * x.sqrt()).collect::<Vec<_>>()
    );

    let idx = Option::<Once<usize>>::None;
    // ASMS figure
    let asm_shape = asms.mirror_shape(idx.clone());
    println!("ASM std: {:.0}nm", 1e9 * asm_shape.std());
    // residual wavefront
    let mut residuals = opd.clone();
    asms.mirror_shape_sub(&mut residuals, idx);
    println!("Residuals std: {:.0}nm", 1e9 * residuals.std());

    let size = (512, 512);
    let _: complot::Heatmap = (
        (opd.map_10e(-6).as_slice(), size),
        complot::complot!("domeseeing-micron.png"),
    )
        .into();
    let _: complot::Heatmap = (
        (asm_shape.map_10e(-6).as_slice(), size),
        complot::complot!("asmshape-micron.png"),
    )
        .into();
    let _: complot::Heatmap = (
        (residuals.map_10e(-9).as_slice(), size),
        complot::complot!("residuals-nm.png"),
    )
        .into();

    Ok(())
}
