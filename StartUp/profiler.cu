#include <iostream>

using namespace std;

__global__ void kernel_call() {}

int main() {

    float *host_in;
    float *dev_in;

    size_t N = 1 << 24;

    cudaEvent_t st1, et1, st2, et2;
    cudaEventCreate(&st1);
    cudaEventCreate(&et1);
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);

    float ms1, ms2;

    // create buffer on host
    host_in = (float *)malloc(N * sizeof(float));

    // create buffer on device
    cudaError_t err = cudaMalloc(&dev_in, N * sizeof(float));
    if (err != cudaSuccess) {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    // record time at start
    cudaEventRecord(st1);
    cudaMemcpy(dev_in, host_in, sizeof(float) * N, cudaMemcpyHostToDevice);

    // no sync required here because memcpy is synchronized
    cudaEventRecord(et1);

    // record time at start
    cudaEventRecord(st2);

    kernel_call<<<4, 1024>>>();

    // wait until kernel is done start timing
    cudaDeviceSynchronize();
    cudaEventRecord(et2);

    cudaEventElapsedTime(&ms1, st1, et1);
    cudaEventElapsedTime(&ms2, st2, et2);

    cout << "MemCpy: " << N << " floats:\t" << ms1 << "ms" << endl;
    cout << "Kernel:\t\t\t" << ms2 << "ms" << endl;

    free(host_in);
    cudaFree(dev_in);

    return 0;
}
