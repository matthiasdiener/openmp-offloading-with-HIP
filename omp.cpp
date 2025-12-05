#include <omp.h>
#include <stdio.h>

void saxpy_omp(float* y, const float* x, float a, int n) {

#pragma omp target
{
    printf("OpenMP offloading available: %s\n", omp_is_initial_device() == 0 ? "true" : "false");
}

#pragma omp target teams distribute parallel for map(to: x[0:n]) map(tofrom: y[0:n])
for (int i = 0; i < n; ++i) {
    y[i] += a * x[i];
}
}
