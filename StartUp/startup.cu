#include <iostream>
#include <iomanip>

using namespace std;

__global__ void kernel_call() {}

/**
 * This code is entirely generated with Gemini 3 Flash
 * I fed it the original startup.cu and just asked it to modify it to iterate through 2^10 to 2^24 in increments of powers of 2
 * It outputs in csv-format for easy use in my python plotters
 */
int main() {
    // Range: 2^10 to 2^24
    const int START_POW = 10;
    const int END_POW = 24;
    
    cudaEvent_t st1, et1, st2, et2;
    cudaEventCreate(&st1);
    cudaEventCreate(&et1);
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);

    // Print CSV header for easy Python plotting
    cout << "N_elements,Memcpy_ms,Kernel_ms" << endl;

    for (int i = START_POW; i <= END_POW; i++) {
        size_t N = 1ULL << i; // Use unsigned long long to avoid overflow at higher powers
        size_t bytes = N * sizeof(float);

        float *host_in, *dev_in;

        // Allocate
        host_in = (float *)malloc(bytes);
        cudaError_t err = cudaMalloc(&dev_in, bytes);
        
        if (err != cudaSuccess) {
            cerr << "Memory allocation failed for N = " << N << endl;
            break; 
        }

        // Benchmark Memcpy (Host to Device)
        cudaEventRecord(st1);
        cudaMemcpy(dev_in, host_in, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(et1);
        cudaEventSynchronize(et1);

        // Benchmark Kernel
        cudaEventRecord(st2);
        kernel_call<<<4, 1024>>>();
        cudaEventRecord(et2);
        cudaEventSynchronize(et2);

        float ms1, ms2;
        cudaEventElapsedTime(&ms1, st1, et1);
        cudaEventElapsedTime(&ms2, st2, et2);

        // Output results in CSV format
        cout << N << "," << ms1 << "," << ms2 << endl;

        // Cleanup this iteration
        free(host_in);
        cudaFree(dev_in);
    }

    // Cleanup events
    cudaEventDestroy(st1);
    cudaEventDestroy(et1);
    cudaEventDestroy(st2);
    cudaEventDestroy(et2);

    return 0;
}