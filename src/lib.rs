#[allow(unused_imports)]
use blst::*;

use core::ffi::c_void;
use std::fmt::Debug;

sppark::cuda_error!();

// #[link(name = "snarkvm_algorithms_cuda")]
extern "C" {
    fn snarkvm_msm(
        ret: *mut c_void,
        points: *const c_void,
        npoints: usize,
        scalars: *const c_void,
        ffi_affine_sz: usize,
    ) -> cuda::Error;
}

/// Compute a multi-scalar multiplication
pub fn sppark_msm<Affine, Projective: Debug, Scalar>(points: &[Affine], scalars: &[Scalar]) -> Result<Projective, cuda::Error> {
    let npoints = scalars.len();
    if npoints > points.len() {
        panic!("length mismatch {} points < {} scalars", npoints, scalars.len())
    }
    #[allow(clippy::uninit_assumed_init)]
    let mut ret: Projective = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let err = unsafe {
        snarkvm_msm(
            &mut ret as *mut _ as *mut c_void,
            points as *const _ as *const c_void,
            npoints,
            scalars as *const _ as *const c_void,
            std::mem::size_of::<Affine>(),
        )
    };
    if err.code != 0 {
        return Err(err);
    }
    Ok(ret)
}