use criterion::{criterion_group, criterion_main, Criterion};

use ark_bls12_377::{Fr,G1Affine,G1Projective};
use ark_ff::BigInteger256;

use std::str::FromStr;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_std::UniformRand;

pub fn generate_points_scalars<G: AffineCurve>(
    len: usize,
) -> (Vec<G>, Vec<G::ScalarField>) {
    let rand_gen: usize = 1 << 11;
    let mut rng = ChaCha20Rng::from_entropy();

    let mut points =
        <G::Projective as ProjectiveCurve>::batch_normalization_into_affine(
            &(0..rand_gen)
                .map(|_| G::Projective::rand(&mut rng))
                .collect::<Vec<_>>(),
        );
    // Sprinkle in some infinity points
    points[3] = G::zero();
    let scalars = (0..len)
        .map(|_| G::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();

    while points.len() < len {
        points.append(&mut points.clone());
    }

    points.truncate(len);

    (points, scalars)
}

fn sppark_msm_benchmark(c: &mut Criterion) {
    let bench_npow = std::env::var("BENCH_NPOW").unwrap_or("13".to_string());
    let npoints_npow = i32::from_str(&bench_npow).unwrap();

    let batches = 1;
    let (points, scalars) = generate_points_scalars::<G1Affine>(1usize << npoints_npow);

    let mut group = c.benchmark_group("cuda");
    group.sample_size(10);

    let name = format!(" msm 2**{} \n", npoints_npow);
    
    group.bench_function(name, |b| {
        b.iter(|| {
            let res = cuda_msm::sppark_msm_gpu::<G1Affine, G1Projective,Fr>(&points[..],&scalars[..]);
            match res {
                Ok(value) => println!("Result: {:?}", value),
                Err(_err) => (),
            }
        })
    });

    group.finish();
}

fn snarkvm_benchmark(c: &mut Criterion) {
    cuda_msm::init_gpu();
    let bench_npow = std::env::var("BENCH_NPOW").unwrap_or("13".to_string());
    let npoints_npow = i32::from_str(&bench_npow).unwrap();

    let batches = 1;
    let (points, scalars) = generate_points_scalars::<G1Affine>(1usize << npoints_npow);

    let mut group = c.benchmark_group("cuda");
    group.sample_size(10);

    let name = format!(" msm 2**{} \n", npoints_npow);
    
    group.bench_function(name, |b| {
        b.iter(|| {
            let res = cuda_msm::msm_gpu::<G1Affine, G1Projective,Fr>(&points[..],&scalars[..]);
            match res {
                Ok(value) => println!("Result: {:?}", value),
                Err(_err) => (),
            }
        })
    });

    group.finish();
    cuda_msm::cleanup_gpu();
}

criterion_group!(benches, snarkvm_benchmark);
criterion_main!(benches);
