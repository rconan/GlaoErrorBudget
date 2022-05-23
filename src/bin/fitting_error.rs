use glao_error_budget::{OpdRecord, OpdStats};
use nalgebra as na;
use parse_monitors::cfd;
use rayon::prelude::*;
use std::fs::File;

pub fn polyfit<T: na::RealField + Copy>(
    x_values: &[T],
    y_values: &[T],
    polynomial_degree: usize,
) -> Result<Vec<T>, &'static str> {
    let number_of_columns = polynomial_degree + 1;
    let number_of_rows = x_values.len();
    let mut a = na::DMatrix::zeros(number_of_rows, number_of_columns);

    for (row, &x) in x_values.iter().enumerate() {
        // First column is always 1
        a[(row, 0)] = T::one();

        for col in 1..number_of_columns {
            a[(row, col)] = x.powf(na::convert(col as f64));
        }
    }

    let b = na::DVector::from_row_slice(y_values);

    let decomp = na::SVD::new(a, true, true);

    match decomp.solve(&b, na::convert(1e-18f64)) {
        Ok(mat) => Ok(mat.data.into()),
        Err(error) => Err(error),
    }
}

fn main() -> anyhow::Result<()> {
    let results: anyhow::Result<Vec<_>> = cfd::Baseline::<2021>::mount()
        .into_iter()
        .map(|cfd_case| cfd_case.to_string())
        .collect::<Vec<String>>()
        .into_par_iter()
        .map(|cfd_case| {
            let path = cfd::Baseline::<2021>::path()
                .join(&cfd_case)
                .join("domeseeing-lstsq_kl.bin");
            let record_file = File::open(path)?;
            let records: Vec<OpdRecord> = bincode::deserialize_from(record_file)?;
            let mean_std = records.mean_std() * 1e9;
            let mean_segment_rss: Vec<_> = records
                .mean_segment_rss()
                .into_iter()
                .map(|x| x * 1e9)
                .collect();
            let mean_modal_coefs_square = records.mean_modal_coefs_square();
            let n_mode = 500;
            let u: Vec<_> = (1..=n_mode).map(|i| (i as f64).ln()).collect();
            let fit: Vec<_> = mean_modal_coefs_square
                .chunks(500)
                .map(|c| {
                    let log_c: Vec<_> = c.iter().map(|&x| x.ln()).collect();
                    polyfit(&u, &log_c, 1).unwrap()
                })
                .collect();
            let eta: Vec<_> = fit.into_iter().map(|x| x[1]).collect();
            let mean_segment_residual_rss = records.mean_segment_residual_rss();
            Ok((
                cfd_case,
                mean_std,
                mean_segment_rss,
                mean_modal_coefs_square,
                eta,
                mean_segment_residual_rss,
            ))
        })
        .collect();
    results.unwrap().into_iter().for_each(
        |(
            cfd_case,
            mean_std,
            mean_segment_rss,
            mean_modal_coefs_square,
            eta,
            mean_segment_residual_rss,
        )| {
            println!("{cfd_case:<20} {mean_std:>6.0} {mean_segment_rss:>6.0?} {eta:>+4.2?}");
            let n_mode = 500;
            {
                let iter = (1..=n_mode).map(|i| {
                    (
                        i as f64,
                        mean_modal_coefs_square
                            .iter()
                            .skip(i - 1)
                            .step_by(n_mode)
                            .take(7)
                            .cloned()
                            .collect::<Vec<f64>>(),
                    )
                });
                let filename = format!("{cfd_case}_modal-spectrum.png");
                let config = complot::Config::new()
                    .filename(filename)
                    .xaxis(complot::Axis::new().label("KL mode # "))
                    .legend(
                        (1..=7)
                            .map(|sid| format!("S{sid}"))
                            .collect::<Vec<String>>(),
                    );
                let _: complot::LogLog = (iter, Some(config)).into();
            }
            {
                let iter = (1..=n_mode).map(|i| {
                    (
                        i as f64,
                        mean_segment_residual_rss
                            .iter()
                            .skip(i - 1)
                            .step_by(n_mode)
                            .take(7)
                            .map(|x| x * 1e9)
                            .collect::<Vec<f64>>(),
                    )
                });
                let filename = format!("{cfd_case}_residuals.png");
                let config = complot::Config::new()
                    .filename(filename)
                    .xaxis(complot::Axis::new().label("KL mode #"))
                    .yaxis(complot::Axis::new().label("Segment WFE RSS [nm]"))
                    .legend(
                        (1..=7)
                            .map(|sid| format!("S{sid}"))
                            .collect::<Vec<String>>(),
                    );
                let _: complot::LinLog = (iter, Some(config)).into();
            }
        },
    );
    Ok(())
}
