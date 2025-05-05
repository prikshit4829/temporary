#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void cudaAddition(int* x, int* y, int* z, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        z[index] = x[index] + y[index];
    }
}

int main() {
    int N = 1e7;
    int *x = new int[N];
    int *y = new int[N];
    int *z = new int[N];

    for (int i = 0; i < N; i++) x[i] = 1, y[i] = 1;

    int *a, *b, *c;
    cudaMalloc(&a, sizeof(int) * N);
    cudaMalloc(&b, sizeof(int) * N);
    cudaMalloc(&c, sizeof(int) * N);

    cudaMemcpy(a, x, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b, y, sizeof(int) * N, cudaMemcpyHostToDevice);

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel with timing
    dim3 th(256);                      // Use 256 threads per block
    dim3 bl((N + th.x - 1) / th.x);    // Compute number of blocks

    cudaEventRecord(start);
    cudaAddition<<<bl, th>>>(a, b, c, N);
    cudaEventRecord(stop);

    // Wait for kernel to finish
    cudaEventSynchronize(stop);

    // Calculate time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time taken by CUDA kernel: " << milliseconds << " ms" << endl;

    cudaMemcpy(z, c, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Optional: Print results (you might want to print only first few for large N)
    // for (int i = 0; i < N; i++) cout << z[i] << " ";

    // Clean up
    delete[] x;
    delete[] y;
    delete[] z;
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
