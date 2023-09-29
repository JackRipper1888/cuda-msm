#include <cuda.h>


#include <ff/bls12-377.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;
#include<msm/pippenger.cuh>

#ifndef __CUDA_ARCH__

extern "C" {

    RustError snarkvm_msm(point_t* out, const affine_t points[], size_t npoints,
                          const scalar_t scalars[], size_t ffi_affine_size) {

        mult_pippenger<bucket_t,point_t,affine_t,scalar_t>(out, points, npoints, scalars,
                                            true, ffi_affine_size);
        return RustError{cudaSuccess};
    }
}

#endif