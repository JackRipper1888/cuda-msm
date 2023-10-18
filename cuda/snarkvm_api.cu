#include <cuda.h>


#include <ff/bls12-377.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef bucket_t::affine_t affine_noinf_t;
typedef fr_t scalar_t;
#include<msm/pippenger.cuh>
#include "../msm-cuda/export.h"
#include "../msm-cuda/MSMParams.hpp"
template<typename T> class host_ptr_t {
    T* h_ptr;
public:
    host_ptr_t(size_t nelems) : h_ptr(nullptr)
    {
        if (nelems) {
            CUDA_OK(cudaMallocHost(&h_ptr, nelems * sizeof(T)));
        }
    }
    ~host_ptr_t() { if (h_ptr) cudaFreeHost((void*)h_ptr); }

    inline operator const T*() const            { return h_ptr; }
    inline operator T*() const                  { return h_ptr; }
    inline operator void*() const               { return (void*)h_ptr; }
    inline const T& operator[](size_t i) const  { return h_ptr[i]; }
    inline T& operator[](size_t i)              { return h_ptr[i]; }
};



#ifndef __CUDA_ARCH__
#include <vector>
#include <chrono>
#include <unistd.h>
#include <atomic>

typedef std::chrono::high_resolution_clock Clock;

using namespace std;

class snarkvm_t {
public:
    size_t max_lg_domain;
    size_t max_lg_blowup;

    struct resource_t {
        int dev;
        int stream;
        resource_t(int _dev, int _stream) {
            dev = _dev;
            stream = _stream;
        }
    };
    channel_t<resource_t*> resources;
    uint32_t num_gpus;

    // Memory will be structured as:
    //     GPU          Host
    //     stream0      stream0
    //       MSM          MSM
    //       Poly         Poly
    //     stream1      stream1
    //       MSM          MSM
    //       Poly         Poly
    //     batch_eval
    std::vector<dev_ptr_t<fr_t>*> d_mem;
    std::vector<host_ptr_t<fr_t>*> h_mem;
    size_t d_msm_elements_per_stream;
    size_t h_msm_elements_per_stream;
    size_t d_poly_elements_per_stream;
    size_t h_poly_elements_per_stream;

    fr_t *d_addr_msm(uint32_t dev, uint32_t stream) {
        fr_t* dmem = *d_mem[dev];
        return &dmem[(d_msm_elements_per_stream + d_poly_elements_per_stream) * stream];
    }
    fr_t *d_addr_poly(uint32_t dev, uint32_t stream) {
        fr_t* dmem = *d_mem[dev];
        return &dmem[(d_msm_elements_per_stream + d_poly_elements_per_stream) * stream +
                     d_msm_elements_per_stream];
    }
    fr_t *h_addr_msm(uint32_t dev, uint32_t stream) {
        fr_t* hmem = *h_mem[dev];
        return &hmem[(h_msm_elements_per_stream + h_poly_elements_per_stream) * stream];
    }
    fr_t *h_addr_poly(uint32_t dev, uint32_t stream) {
        fr_t* hmem = *h_mem[dev];
        return &hmem[(h_msm_elements_per_stream + h_poly_elements_per_stream) * stream +
                     h_msm_elements_per_stream];
    }

    // Cache for MSM bases.
    static const size_t msm_cache_entries = 4;
    // Key to cached msm_t mapping
    uint64_t msm_keys[msm_cache_entries];
    // Max number of points to support
    static const size_t msm_cache_npoints = 65537;
    // Current cache entry
    size_t cur_msm_cache_entry;
    struct dev_msm_point_cache_t {
        dev_ptr_t<affine_noinf_t>* cache[msm_cache_entries];
    };
    std::vector<dev_msm_point_cache_t> msm_point_cache;
    

    // Cache batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs_over_domain values
    size_t batch_eval_ratio = 4; // mul_domain / constraint_domain
    size_t d_batch_eval_domain = 131072;
    std::vector<fr_t*> d_batch_eval;

    // MSM kernel context, per device and stream
    struct msm_cuda_ctx_t {
        void* ctxs[gpu_t::FLIP_FLOP];
    };
    std::vector<msm_cuda_ctx_t> msm_cuda_ctxs;


    snarkvm_t() {
        max_lg_domain = 17;
        size_t domain_size = (size_t)1 << max_lg_domain;
        size_t ext_domain_size = domain_size;

        // Set up MSM kernel context
        // Set up polynomial division kernel
        num_gpus = 1;
        msm_cuda_ctxs.resize(num_gpus);
        for (size_t j = 0; j < gpu_t::FLIP_FLOP; j++) {
            for (size_t dev = 0; dev < num_gpus; dev++) {
                auto &gpu = select_gpu(dev);
                msm_cuda_ctxs[dev].ctxs[j] = msm_cuda_create_context(msm_cache_npoints, gpu.sm_count());
                resources.send(new resource_t(dev, j));
            }
        }

        // The msm library needs storage internally:
        //   gpu - buckets + partial sums
        //   cpu - partial sums
        size_t msm_cuda_host_bytes;
        size_t msm_cuda_gpu_bytes;
        msm_cuda_storage(msm_cuda_ctxs[0].ctxs[0], &msm_cuda_host_bytes, &msm_cuda_gpu_bytes);

        // For MSM we need additional space for scalars. Points are cached.
        msm_cuda_gpu_bytes += msm_cache_npoints * sizeof(fr_t);
        msm_cuda_host_bytes += msm_cache_npoints * (sizeof(fr_t) + sizeof(affine_t));

        // Ensure GPU/CPU storage for either kernel.
        size_t msm_gpu_elements = (msm_cuda_gpu_bytes + sizeof(fr_t) - 1) / sizeof(fr_t);
        size_t msm_host_elements = (msm_cuda_host_bytes + sizeof(fr_t) - 1) / sizeof(fr_t);
        
        // Determine storage needed for polynomial operations
        size_t num_elements = ext_domain_size + domain_size;
        // Storage to operate on up to 5 polynomials at a time
        num_elements *= 5;
        
        d_poly_elements_per_stream = num_elements;
        h_poly_elements_per_stream = num_elements;
        d_msm_elements_per_stream = msm_gpu_elements;
        h_msm_elements_per_stream = msm_host_elements;

        size_t elements_per_stream_gpu = d_poly_elements_per_stream + d_msm_elements_per_stream;
        size_t elements_per_stream_cpu = h_poly_elements_per_stream + h_msm_elements_per_stream;

        // Determine storage needed per device, and host side per device
        size_t elements_per_dev = elements_per_stream_gpu * gpu_t::FLIP_FLOP;
        size_t elements_per_dev_cpu = elements_per_stream_cpu * gpu_t::FLIP_FLOP;
        // Add batch eval values on top since we need to preserve those (per device)
        elements_per_dev += batch_eval_ratio;
        
        d_mem.resize(num_gpus);
        h_mem.resize(num_gpus);
        d_batch_eval.resize(num_gpus);

        for (size_t dev = 0; dev < num_gpus; dev++) {
            auto &gpu = select_gpu(dev);
            d_mem[dev] = new dev_ptr_t<fr_t>(elements_per_dev);
            h_mem[dev] = new host_ptr_t<fr_t>(elements_per_dev_cpu);
            // Batch evals sit at the end of the memory allocation
            d_batch_eval[dev] = &(*d_mem[dev])[elements_per_dev - batch_eval_ratio];
            gpu.sync();
        }

        // Set up the MSM cache
        cur_msm_cache_entry = 0;
        msm_point_cache.resize(num_gpus);
        for (size_t i = 0; i < num_gpus; i++) {
            for (size_t j = 0; j < msm_cache_entries; j++) {
                msm_point_cache[i].cache[j] = nullptr;
            }
        }
        
        // For batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs_over_domain,
        // the pattern of subtractions depends the ratio of the domain sizes
        // ratio = 4
        // pow((g**4)%p, 131072//ratio, p)-1
        fr_t g("e805156baeaf0c35750f3f45a4e8d566148cb16a8de9973f8522249260678ff");
        vector<fr_t> evals;
        evals.resize(batch_eval_ratio);
        // Populate initial values
        evals[0] = fr_t::one();
        evals[1] = g;
        for (size_t i = 2; i < batch_eval_ratio; i++) {
            evals[i] = evals[i - 1] * g;
        }
        // Exponentiate
        fr_t one = fr_t::one();
        for (size_t i = 0; i < batch_eval_ratio; i++) {
            evals[i] = (evals[i] ^ (d_batch_eval_domain / 4)) - one;
        }
        for (size_t dev = 0; dev < num_gpus; dev++) {
            auto &gpu = select_gpu(dev);
            gpu.HtoD(d_batch_eval[dev], &evals[0], batch_eval_ratio);
            gpu.sync();
        }
    }
    ~snarkvm_t() {
        for (size_t dev = 0; dev < num_gpus; dev++) {
            select_gpu(dev);
            delete d_mem[dev];
            delete h_mem[dev];

            // Free MSM caches
            for (size_t i = 0; i < cur_msm_cache_entry; i++) {
                delete msm_point_cache[dev].cache[i];
            }
            //msm_cuda_delete_context(msm_cuda_ctx);
        }
    }


    static const size_t FP_BYTES = 48;
   
    struct rust_p1_affine_t {
        uint8_t x[FP_BYTES];
        uint8_t y[FP_BYTES];
        uint8_t inf;
        uint8_t pad[7];
    };

public:
    void populate_msm_points(int dev, size_t npoints, size_t max_cache,
                             const rust_p1_affine_t* points,
                             affine_noinf_t *h_points, affine_noinf_t *d_points,
                             size_t ffi_affine_sz) {
        auto& gpu = select_gpu(dev);
        // Copy bases, omitting infinity
        assert(sizeof(rust_p1_affine_t) == ffi_affine_sz);
        for (unsigned i = 0; i < npoints; i++) {
            memcpy((uint8_t*)&h_points[i], (uint8_t*)&points[i], sizeof(affine_noinf_t));
        }
        gpu.HtoD(d_points, h_points, max_cache);
        gpu.sync();

        msm_cuda_precompute_bases(msm_cuda_ctxs[dev].ctxs[0], max_cache, d_points, d_points);
    }
    RustError MSMCacheBases(const affine_t points[], size_t bases_len, size_t ffi_affine_sz) {
        uint64_t key = ((uint64_t *)points)[0];
        assert (cur_msm_cache_entry < msm_cache_entries);
        assert(bases_len <= msm_cache_npoints);
        for (size_t devt = 0; devt < num_gpus; devt++) {
            auto& gpu = select_gpu(devt);
            uint32_t windowBits;
            uint32_t allWindows;
            msm_cuda_precomp_params(msm_cuda_ctxs[devt].ctxs[0], &windowBits, &allWindows);

            // Allocate the max cache size, though we might not use it all. This
            // simplifies precomputation on the GPU since they can all be the same
            // size and stride. 
            msm_point_cache[devt].cache[cur_msm_cache_entry] =
                new dev_ptr_t<affine_noinf_t>(msm_cache_npoints * allWindows);
            fr_t* h_buf = *h_mem[devt];
            // Cache the entire set of bases
            populate_msm_points(devt, bases_len, msm_cache_npoints, (const rust_p1_affine_t*)points,
                                (affine_noinf_t*)h_buf, // host buffer
                                *msm_point_cache[devt].cache[cur_msm_cache_entry], // device buffer
                                ffi_affine_sz);

            // Ensure all transfers are complete. Could remove this if precomp used the right streams
            cudaDeviceSynchronize();
        }
        // Store the key
        msm_keys[cur_msm_cache_entry] = key;

        cur_msm_cache_entry++;
        return RustError{cudaSuccess};
    }

    RustError MSM(point_t* out, const affine_t points[],
                  size_t npoints, size_t bases_len,
                  const scalar_t scalars[], size_t ffi_affine_sz)
    {
        RustError result;
        
        resource_t *resource = resources.recv();
        int dev = resource->dev;
        auto& gpu = select_gpu(dev);
        int stream_idx = resource->stream;
        stream_t& stream = gpu[stream_idx];

        // See if these bases are cached
        uint64_t key = ((uint64_t *)points)[0];
        dev_ptr_t<affine_noinf_t>* cached_points = nullptr;
        for (size_t i = 0; i < msm_cache_entries && i < cur_msm_cache_entry; i++) {
            if (key == msm_keys[i]) {
                cached_points = msm_point_cache[dev].cache[i];
                break;
            }
        }
        // Create a new cached msm_t
        // Not MT safe - if we populate the cache from a single thread
        //       prior to going MT this will be ok.
        if (cached_points == nullptr && cur_msm_cache_entry < msm_cache_entries) {
            MSMCacheBases(points, bases_len, ffi_affine_sz);
            // Re-select the target gpu
            select_gpu(dev);
            // And the points cached for the kernel
            cached_points = msm_point_cache[dev].cache[cur_msm_cache_entry - 1];
        }
        assert (cached_points != nullptr);

        fr_t* h_scalars = h_addr_msm(dev, stream_idx);
        memcpy(h_scalars, scalars, sizeof(fr_t) * npoints);
            
        fr_t* d_scalars = d_addr_msm(dev, stream_idx);
        // Must have room for scalars, buckets, bucketsums
        bucket_t* d_buckets = (bucket_t*)&d_scalars[npoints];
        bucket_t* h_bucketSums = (bucket_t*)h_scalars;
        stream.HtoD(d_scalars, h_scalars, npoints);
        affine_noinf_t* cached_points_ptr = *cached_points;

        msm_cuda_launch(msm_cuda_ctxs[dev].ctxs[stream_idx], npoints, out,
                        d_scalars, cached_points_ptr,
                        d_buckets, h_bucketSums,
                        false, stream);

        result = RustError{cudaSuccess};

        resources.send(resource);
        return result;
    }
};

bool amlive = true;
static void shutdown() {
    amlive = false;
}
static bool alive() {
    return amlive;
}

snarkvm_t *snarkvm = nullptr;

extern "C" {
    void snarkvm_init_gpu() {
        snarkvm = new snarkvm_t();
        assert(snarkvm);
    }

    void snarkvm_cleanup_gpu() {
        shutdown();
        sleep(1); // Shutting down - delay while process exits
    }
    
    void* snarkvm_alloc_pinned(size_t bytes) {
        void* ptr = nullptr;
        if (cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault) != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    void snarkvm_free_pinned(void *ptr) {
        cudaFreeHost(ptr);
    }

    RustError snarkvm_msm(point_t* out, const affine_t points[], size_t npoints, size_t bases_len,
                          const scalar_t scalars[], size_t ffi_affine_size) {
        if (!alive()) {
            sleep(10); // Shutting down - delay while process exits
            return RustError{0};
        }
        
        RustError err = RustError{0};
        try {
            err =snarkvm->MSM(out, points, npoints, bases_len, scalars, ffi_affine_size);
        } catch(exception &exc) {
            shutdown();
            sleep(10); // Shutting down - delay while process exits
        }
        return err;
    }

    RustError sppark_msm(point_t* out, const affine_t points[], size_t npoints,
                          const scalar_t scalars[], size_t ffi_affine_size) {

        mult_pippenger<bucket_t,point_t,affine_t,scalar_t>(out, points, npoints, scalars,
                                            true, ffi_affine_size);
        return RustError{cudaSuccess};
    }
}

#endif