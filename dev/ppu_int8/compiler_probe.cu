#include <cuda_fp16.h>

__global__ void compiler_probe(__half *out) {
  int const i = int(blockIdx.x * blockDim.x + threadIdx.x);
  out[i] = __hadd(__float2half(float(i)), __float2half(1.0f));
}

int main() { return 0; }
