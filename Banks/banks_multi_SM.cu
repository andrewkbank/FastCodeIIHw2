#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
/**
 * Code written almost entirely with Gemini 3.0 flash
 * It runs the kernel on multiple SMs for a matrix that is larger than 64x64
 * It is pretty good
 */
__global__ void kernel_call(int N, float *in, float *out) {
    __shared__ float share_buf[64 * 64];

    // Identify which 64x64 tile this block is processing
    // For a transpose, block (bx, by) reads from in(bx, by) and writes to out(by, bx)
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Offsets in the global linear memory
    int in_tile_offset = (by * 64) * N + (bx * 64);
    int out_tile_offset = (bx * 64) * N + (by * 64);

    int id = threadIdx.x; 

    // 1. Read from Global (Coalesced) -> Write to Shared (Skewed)
    for (int i = 0; i < (64 * 64 / blockDim.x); ++i) {
        int linear_idx = i * blockDim.x + id;
        int x_in = linear_idx % 64;
        int y_in = linear_idx / 64;

        // Apply skewed mapping to avoid bank conflicts
        int x_inter = (y_in + x_in) % 64;
        int y_inter = x_in;

        // Read from global memory using the tile offset
        // (y_in * N) moves down rows in the large matrix
        share_buf[y_inter * 64 + x_inter] = in[in_tile_offset + y_in * N + x_in];
    }

    __syncthreads();

    // 2. Read from Shared (Skewed) -> Write to Global (Coalesced)
    for (int i = 0; i < (64 * 64 / blockDim.x); ++i) {
        int linear_idx = i * blockDim.x + id;
        int x_out = linear_idx % 64;
        int y_out = linear_idx / 64;

        int y_inter = y_out;
        int x_inter = (x_out + y_out) % 64;

        // Write to global memory at the TRANSPOSED tile offset
        out[out_tile_offset + y_out * N + x_out] = share_buf[y_inter * 64 + x_inter];
    }
}

int main() {
    float *host_in, *host_out;
    float *dev_in, *dev_out;

    size_t B = 1;
    // Example: A 1024x1024 matrix (must be multiple of 64)
    size_t N = 4096;

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

    // Define Grid and Block dimensions
    // 128 threads per block as requested
    dim3 threadsPerBlock(128); 
    
    // We need enough blocks to cover the N/64 x N/64 grid
    dim3 numBlocks(N / 64, N / 64); 

    cudaEventRecord(st2);
    kernel_call<<<numBlocks, threadsPerBlock>>>(N, dev_in, dev_out);
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
