#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {   //private namespace

constexpr int kThreads = 256;

template <typename scalar_t>   //the same kernal can be applied to different data types (float, double, half)

__global__ void gradient_kernel(

    const scalar_t* __restrict__ input,    //input is  a pointer to a scalar_t type
    scalar_t* __restrict__ grad_h,         //grad_h is a pointer to a scalar_t type
    scalar_t* __restrict__ grad_v,      //grad_v is a pointer to a scalar_t type
    int64_t elements,
    int64_t height,
    int64_t width

  ) 
 
  {

  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;

  const int64_t x = index % width;
  const int64_t y = (index / width) % height;
  const int64_t plane_start = index - y * width - x;
  const int64_t right = plane_start + y * width + ((x + 1) % width);
  const int64_t down = plane_start + ((y + 1) % height) * width + x;

  grad_h[index] = input[right] - input[index];
  grad_v[index] = input[down] - input[index];
}

template <typename scalar_t>
__global__ void divergence_kernel(
    const scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ grad_v,
    scalar_t* __restrict__ output,
    int64_t elements,
    int64_t height,
    int64_t width) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;

  const int64_t x = index % width;
  const int64_t y = (index / width) % height;
  const int64_t plane_start = index - y * width - x;
  const int64_t left = plane_start + y * width + ((x + width - 1) % width);
  const int64_t up = plane_start + ((y + height - 1) % height) * width + x;

  output[index] = grad_h[left] - grad_h[index] +
                  grad_v[up] - grad_v[index];
}

template <typename scalar_t>
__global__ void shrink_dual_kernel(
    const scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ grad_v,
    const scalar_t* __restrict__ dual_h,
    const scalar_t* __restrict__ dual_v,
    const scalar_t* __restrict__ threshold,
    bool scalar_threshold,
    scalar_t epsilon,
    scalar_t* __restrict__ split_h,
    scalar_t* __restrict__ split_v,
    scalar_t* __restrict__ next_dual_h,
    scalar_t* __restrict__ next_dual_v,
    int64_t elements) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;

  const scalar_t q_h = grad_h[index] + dual_h[index];
  const scalar_t q_v = grad_v[index] + dual_v[index];
  const scalar_t norm = sqrt(q_h * q_h + q_v * q_v);
  const scalar_t tau = threshold[scalar_threshold ? 0 : index];
  const scalar_t denominator = norm > epsilon ? norm : epsilon;
  const scalar_t scale = norm > tau ? (norm - tau) / denominator : scalar_t(0);
  const scalar_t t_h = scale * q_h;
  const scalar_t t_v = scale * q_v;

  split_h[index] = t_h;
  split_v[index] = t_v;
  next_dual_h[index] = dual_h[index] + grad_h[index] - t_h;
  next_dual_v[index] = dual_v[index] + grad_v[index] - t_v;
}

}  // namespace





std::vector<torch::Tensor> admm_gradient_cuda(torch::Tensor input) {
  
  const c10::cuda::CUDAGuard device_guard(input.device());   //select the correct GPU device for the operation
  
  auto grad_h = torch::empty_like(input);
  auto grad_v = torch::empty_like(input);
  
  const int64_t elements = input.numel();
  const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "admm_gradient_cuda", [&] {
  
  gradient_kernel<scalar_t><<<blocks, kThreads, 0,c10::cuda::getCurrentCUDAStream()>>>(input.data_ptr<scalar_t>(), grad_h.data_ptr<scalar_t>(), grad_v.data_ptr<scalar_t>(), elements, input.size(-2), input.size(-1));});

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  
  return {grad_h, grad_v};

}

torch::Tensor admm_divergence_cuda(torch::Tensor grad_h, torch::Tensor grad_v) {
  const c10::cuda::CUDAGuard device_guard(grad_h.device());
  auto output = torch::empty_like(grad_h);
  const int64_t elements = grad_h.numel();
  const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_h.scalar_type(), "admm_divergence_cuda", [&] {
    divergence_kernel<scalar_t><<<blocks, kThreads, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        grad_h.data_ptr<scalar_t>(), grad_v.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), elements, grad_h.size(-2), grad_h.size(-1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> admm_shrink_dual_cuda(
    torch::Tensor grad_h,
    torch::Tensor grad_v,
    torch::Tensor dual_h,
    torch::Tensor dual_v,
    torch::Tensor threshold,
    double epsilon) {
  const c10::cuda::CUDAGuard device_guard(grad_h.device());
  threshold = threshold.to(grad_h.options()).contiguous();
  auto split_h = torch::empty_like(grad_h);
  auto split_v = torch::empty_like(grad_v);
  auto next_dual_h = torch::empty_like(dual_h);
  auto next_dual_v = torch::empty_like(dual_v);
  const int64_t elements = grad_h.numel();
  const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
  const bool scalar_threshold = threshold.numel() == 1;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_h.scalar_type(), "admm_shrink_dual_cuda", [&] {
    shrink_dual_kernel<scalar_t><<<blocks, kThreads, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        grad_h.data_ptr<scalar_t>(), grad_v.data_ptr<scalar_t>(),
        dual_h.data_ptr<scalar_t>(), dual_v.data_ptr<scalar_t>(),
        threshold.data_ptr<scalar_t>(), scalar_threshold,
        static_cast<scalar_t>(epsilon), split_h.data_ptr<scalar_t>(),
        split_v.data_ptr<scalar_t>(), next_dual_h.data_ptr<scalar_t>(),
        next_dual_v.data_ptr<scalar_t>(), elements);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {split_h, split_v, next_dual_h, next_dual_v};
}
