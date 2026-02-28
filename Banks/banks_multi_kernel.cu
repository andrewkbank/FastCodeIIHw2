#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

// This kernel only processes ONE 64x64 tile based on the offsets provided
__global__ void kernel_tile_transpose(int N, float *in, float *out, int tile_x,
                                      int tile_y) {
    __shared__ float share_buf[64 * 64];

    // Offsets in the global linear memory for this specific tile
    int in_offset = (tile_y * 64) * N + (tile_x * 64);
    int out_offset = (tile_x * 64) * N + (tile_y * 64);

    int id = threadIdx.x;

    // 1. Load Tile (Coalesced + Skewed)
    for (int i = 0; i < 32; ++i) {
        int linear_idx = i * blockDim.x + id;
        int x_in = linear_idx % 64;
        int y_in = linear_idx / 64;

        int x_inter = (y_in + x_in) % 64;
        int y_inter = x_in;

        share_buf[y_inter * 64 + x_inter] = in[in_offset + y_in * N + x_in];
    }

    __syncthreads();

    // 2. Store Tile (Skewed -> Coalesced)
    for (int i = 0; i < 32; ++i) {
        int linear_idx = i * blockDim.x + id;
        int x_out = linear_idx % 64;
        int y_out = linear_idx / 64;

        int y_inter = y_out;
        int x_inter = (x_out + y_out) % 64;

        out[out_offset + y_out * N + x_out] = share_buf[y_inter * 64 + x_inter];
    }
}

int main() {
    float *host_in, *host_out;
    float *dev_in, *dev_out;

    size_t B = 1;
    // Example: A 1024x1024 matrix (must be multiple of 64)
    size_t N = 4096;
    size_t num_tiles_per_dim = N / 64;

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

    // Launch a NEW kernel for every single tile in the matrix
    for (int ty = 0; ty < num_tiles_per_dim; ++ty) {
        for (int tx = 0; tx < num_tiles_per_dim; ++tx) {
            // Launch exactly 1 block.
            // Because only 1 block is launched, it only occupies 1 SM at a
            // time.
            kernel_tile_transpose<<<1, 128>>>(N, dev_in, dev_out, tx, ty);
        }
    }

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
