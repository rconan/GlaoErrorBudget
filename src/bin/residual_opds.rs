use glao_error_budget::{ASM, ASMS, OPD};
use parse_monitors::cfd;

fn main() -> anyhow::Result<()> {
    let mut asms: Vec<ASM> = ASMS::from_bins()?;
    let results: anyhow::Result<Vec<_>> = cfd::Baseline::<2021>::mount()
        .into_iter()
        .map(|cfd_case| {
            let files: Vec<_> = cfd::CfdDataFile::<2021>::OpticalPathDifference
                .glob(cfd_case)
                .unwrap()
                .collect();
            (
                cfd_case.to_string(),
                files
                    .last()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string(),
            )
        })
        .map(|(cfd_case, last_opd_file)| {
            println!("CFD case: {cfd_case}");
            let opd = OPD::from_npz(last_opd_file)?;
            let size = (512, 512);
            let filename = format!("{cfd_case}_domeseeing-micron.png");
            let _: complot::Heatmap = (
                (opd.map_10e(-6).as_slice(), size),
                complot::complot!(filename),
            )
                .into();
            //asms.project(&opd)?;
            asms.least_square(&opd)?;
            let residuals = &opd - &asms;
            let filename = format!("{cfd_case}_residuals-opd.png");
            let _: complot::Heatmap = (
                (residuals.map_10e(-9).as_slice(), size),
                complot::complot!(filename),
            )
                .into();
            Ok(())
        })
        .collect();
    Ok(())
}
