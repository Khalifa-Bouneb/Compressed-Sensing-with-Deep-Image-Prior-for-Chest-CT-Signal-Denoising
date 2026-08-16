#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
//(torch::tensor input) INPUT
//(std::vector<torch::Tensor>) OUTPUT == VECTOR OF TENSORS
std::vector<torch::Tensor> admm_gradient_cuda(torch::Tensor input);

torch::Tensor admm_divergence_cuda(torch::Tensor grad_h, torch::Tensor grad_v);

std::vector<torch::Tensor> admm_shrink_dual_cuda(
    torch::Tensor grad_h,
    torch::Tensor grad_v,
    torch::Tensor dual_h,
    torch::Tensor dual_v,
    torch::Tensor threshold,
    double epsilon);

std::vector<torch::Tensor> admm_gradient(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "admm_gradient expects a CUDA tensor");
  TORCH_CHECK(input.dim() == 4, "input must have NCHW layout");
  return admm_gradient_cuda(input.contiguous());
}

torch::Tensor admm_divergence(torch::Tensor grad_h, torch::Tensor grad_v) {
  TORCH_CHECK(grad_h.is_cuda() && grad_v.is_cuda(),
              "admm_divergence expects CUDA tensors");
  TORCH_CHECK(grad_h.sizes() == grad_v.sizes(),
              "horizontal and vertical gradients must have identical shapes");
  return admm_divergence_cuda(grad_h.contiguous(), grad_v.contiguous());
}

std::vector<torch::Tensor> admm_shrink_dual(
    torch::Tensor grad_h,
    torch::Tensor grad_v,
    torch::Tensor dual_h,
    torch::Tensor dual_v,
    torch::Tensor threshold,
    double epsilon) {
  TORCH_CHECK(grad_h.is_cuda() && grad_v.is_cuda() && dual_h.is_cuda() &&
                  dual_v.is_cuda() && threshold.is_cuda(),
              "admm_shrink_dual expects CUDA tensors");
  TORCH_CHECK(grad_h.sizes() == grad_v.sizes() &&
                  grad_h.sizes() == dual_h.sizes() &&
                  grad_h.sizes() == dual_v.sizes(),
              "gradient and dual tensors must have identical shapes");
  TORCH_CHECK(threshold.numel() == 1 || threshold.sizes() == grad_h.sizes(),
              "threshold must be scalar or have the same shape as gradients");
  return admm_shrink_dual_cuda(
      grad_h.contiguous(), grad_v.contiguous(), dual_h.contiguous(),
      dual_v.contiguous(), threshold.contiguous(), epsilon);
}

//this will map the C++ kernel to python funtion :
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {

  module.def("gradient", &admm_gradient, "Periodic forward differences (CUDA)");
  module.def("divergence", &admm_divergence, "Adjoint periodic differences (CUDA)");
  module.def("shrink_dual", &admm_shrink_dual, "Fused isotropic shrinkage and ADMM dual update (CUDA)");

}
