#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void smooth_l1_kernel_cuda(TensorIterator& iter, double beta, bool huber) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "smooth_l1_cuda", [&iter, beta, huber]() {
    scalar_t beta_val(beta);
    gpu_kernel(iter, [beta_val, huber]GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      auto z = ::abs(a - b);
      if (z < beta_val) {
        const auto output = scalar_t(0.5) * z * z;
        return huber ? output : (output / beta_val);
      } else {
        const auto output = z - scalar_t(0.5) * beta_val;
        return huber ? (output * beta_val) : output;
      }
    });
  });
}


void mse_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "mse_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      auto diff = a - b;
      return diff * diff;
    });
  });
}

REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel_cuda);
REGISTER_DISPATCH(mse_stub, &mse_kernel_cuda);

// DO NOT ADD ANY NEW KERNELS HERE
// CUDA compilation times grow quickly.  It's perfectly acceptable to have a file per kernel.

}} // namespace at::native
