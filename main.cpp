#include <iostream>
#include <vector>

void saxpy_hip(float* y, const float* x, float a, int n);
void saxpy_omp(float* y, const float* x, float a, int n);

int main() {
    const int n = 1024;
    std::vector<float> x(n, 42.0f);
    std::vector<float> y_hip(n, 0.0f);
    std::vector<float> y_omp(n, 0.0f);

    saxpy_hip(y_hip.data(), x.data(), 2.0f, n);
    saxpy_omp(y_omp.data(), x.data(), 2.0f, n);

    std::cout << "HIP  y[0] = " << y_hip[0] << "\n";
    std::cout << "OpenMP y[0] = " << y_omp[0] << "\n";

    return 0;
}
