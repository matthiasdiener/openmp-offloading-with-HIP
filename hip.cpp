// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

__global__ void saxpy_kernel(float* __restrict__ y, const float* __restrict__ x,    float a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += a * x[i];
    }
}


void saxpy_hip(float* y, const float* x, float a, int n)
{
    float* d_x = nullptr;
    float* d_y = nullptr;

    (void) hipMalloc(&d_x, n * sizeof(float));
    (void) hipMalloc(&d_y, n * sizeof(float));

    (void) hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
    (void) hipMemcpy(d_y, y, n * sizeof(float), hipMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    hipLaunchKernelGGL(saxpy_kernel, grid, block, 0, 0, d_y, d_x, a, n);
    (void) hipDeviceSynchronize();

    (void) hipMemcpy(y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    (void) hipFree(d_x);
    (void) hipFree(d_y);
}
