#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

__global__ void kernel_call(int N, float *in, float *out) {
    __shared__ float share_buf[64 * 64];

    // DO NOT CHANGE ANY CODE ABOVE THIS COMMENT

    int id = threadIdx.x; // get my id;

    // 1. Read from Global (Coalesced) -> Write to Shared (Skewed)
    for (int i = 0; i < (64 * 64 / blockDim.x); ++i) {
        int linear_idx = i * blockDim.x + id;

        // Convert linear index to 2D (x_in, y_in)
        int x_in = linear_idx % 64;
        int y_in = linear_idx / 64;

        // Apply formula: (x_in, y_in) -> ((y_in + x_in) % 64, x_in)
        int x_inter = (y_in + x_in) % 64;
        int y_inter = x_in;

        share_buf[y_inter * 64 + x_inter] = in[linear_idx];
    }

    __syncthreads();

    // 2. Read from Shared (Skewed) -> Write to Global (Coalesced)
    for (int i = 0; i < (64 * 64 / blockDim.x); ++i) {
        int linear_idx = i * blockDim.x + id;

        // For the output, we treat linear_idx as the (x_out, y_out)
        int x_out = linear_idx % 64;
        int y_out = linear_idx / 64;

        // To get the transpose, we need the original (y_in, x_in)
        // From your formula: x_intermediate = (y_in + x_in) % 64,
        // y_intermediate = x_in Since we want the transpose: y_intermediate =
        // y_out, x_intermediate = (x_out + y_out) % 64
        int y_inter = y_out;
        int x_inter = (x_out + y_out) % 64;

        out[linear_idx] = share_buf[y_inter * 64 + x_inter];
    }
}

int main() {
    float *host_in, *host_out;
    float *dev_in, *dev_out;

    size_t B = 1;
    size_t N = 64;

    // create buffer on host
    host_in = (float *)malloc(B * B * N * N * sizeof(float));
    host_out = (float *)malloc(B * B * N * N * sizeof(float));

    // creates a matrix stored in row major order
    for (int ii = 0; ii < B; ++ii) {
        for (int jj = 0; jj < B; ++jj) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    host_in[(ii * B + jj) * N * N + i * N + j] = i * N + j;
                }
            }
        }
    }

    // create buffer on device
    cudaError_t err = cudaMalloc(&dev_in, B * B * N * N * sizeof(float));
    if (err != cudaSuccess) {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    err = cudaMalloc(&dev_out, B * B * N * N * sizeof(float));
    if (err != cudaSuccess) {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    cudaMemcpy(dev_in, host_in, B * B * N * N * sizeof(float),
               cudaMemcpyHostToDevice);

    // create GPU timing events for timing the GPU
    cudaEvent_t st2, et2;
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);

    cudaEventRecord(st2);
    kernel_call<<<1, 128>>>(N, dev_in, dev_out);
    cudaEventRecord(et2);

    // host waits until et2 has occured
    cudaEventSynchronize(et2);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st2, et2);

    cout << "Kernel time: " << milliseconds << "ms" << endl;

    // copy data out
    cudaMemcpy(host_out, dev_out, B * B * N * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int ii = 0; ii < B; ++ii) {
        for (int jj = 0; jj < B; ++jj) {
            for (int i = 0; i != N; ++i) {
                for (int j = 0; j != N; ++j) {
                    correct &= (host_out[(ii * B + jj) * N * N + i * N + j] ==
                                host_in[(jj * B + ii) * N * N + j * N + i]);
                }
            }
        }
    }
    cout << (correct ? "Yes" : "No") << endl;

    cudaEventDestroy(st2);
    cudaEventDestroy(et2);

    free(host_in);
    free(host_out);
    cudaFree(dev_in);
    cudaFree(dev_out);

    return 0;
}
