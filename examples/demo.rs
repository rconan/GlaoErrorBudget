use glao_error_budget::{ASM, ASMS, OPD};
use std::{iter::Once, time::Instant};

fn main() -> anyhow::Result<()> {
    // Loading the OPD
    let opd = OPD::from_npz("optvol_optvol_6.000000e+02.npz")?;
    println!("{}/{}", opd.no_nan_opd().count(), 512 * 512);
    println!("OPD mean: {:.0}nm", 1e9 * opd.mean());
    println!("OPD std: {:.0}nm", 1e9 * opd.std());

    // Collecting the 7 modal basis
    println!("Assembling the ASM segments ...");
    let now = Instant::now();
    let mut asms: Vec<ASM> = ASMS::from_bins()?;
    println!(" done in {}s", now.elapsed().as_secs());

    // OPD projection
    println!("Projecting the opd on the ASM segments ...");
    let now = Instant::now();
    asms.project(&opd)?;
    println!(" done in {}ms", now.elapsed().as_millis());

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
    println!(
        "Segment Residual WFE RMS: {:.0?}nm",
        vars.iter()
            .zip(&stds)
            .map(|(x, y)| (y * y - x * 1e18).abs().sqrt())
            .collect::<Vec<_>>()
    );
    let n_points: Vec<_> = asms.iter().map(|asm| asm.n_point()).collect();
    let nn_points: usize = n_points.iter().sum();
    let c: Vec<_> = n_points
        .into_iter()
        .map(|x| x as f64 / nn_points as f64)
        .collect();
    println!(
        "Residual WFE RMS: {:.0?}nm",
        (vars
            .iter()
            .zip(&stds)
            .map(|(x, y)| (y * y - x * 1e18).abs())
            .zip(&c)
            .map(|(x, c)| x * c)
            .sum::<f64>())
        .sqrt()
    );

    let idx = Option::<Once<usize>>::None;
    // ASMS figure
    let asm_shape = asms.mirror_shape(idx);
    println!("ASM std: {:.0}nm", 1e9 * asm_shape.std());
    // Residual wavefront
    let residuals = &opd - &asms;
    println!("Residuals std: {:.0}nm", 1e9 * residuals.std());

    let stds: Vec<_> = asms
        .iter()
        .map(|asm| 1e9 * residuals.masked_rms(asm.mask()))
        .collect();
    println!("Segment residuals STD: {:.0?}nm", stds);
    println!(
        "Segment residuals RMSS: {:.0?}nm",
        (stds.iter().map(|x| x * x).sum::<f64>() / 7f64).sqrt()
    );

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
