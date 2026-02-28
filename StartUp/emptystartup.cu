#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>

using namespace std;

// A truly empty kernel to measure pure overhead
__global__ void empty_kernel() { }


/**
 * This code is entirely generated with Gemini 3 Flash
 * I simply asked it to generate me similar code to startup.cu, except only run on empty kernals and compile the average startup time.
 */
int main() {
    const int WARMUP_ITERATIONS = 10;
    const int TEST_ITERATIONS = 1000;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. Warm-up phase
    // GPUs often downclock when idle. We run a few kernels to "wake up" the hardware.
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        empty_kernel<<<1, 1>>>();
    }
    cudaDeviceSynchronize();

    vector<float> latencies;
    latencies.reserve(TEST_ITERATIONS);

    // 2. Benchmarking phase
    for (int i = 0; i < TEST_ITERATIONS; ++i) {
        cudaEventRecord(start);
        
        // Launching a minimal configuration
        empty_kernel<<<1, 1>>>();
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // Wait for this specific launch to finish

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        latencies.push_back(ms);
    }

    // 3. Statistical Analysis
    float sum = accumulate(latencies.begin(), latencies.end(), 0.0f);
    float avg = sum / TEST_ITERATIONS;

    float sq_sum = inner_product(latencies.begin(), latencies.end(), latencies.begin(), 0.0f);
    float stdev = sqrt(sq_sum / TEST_ITERATIONS - avg * avg);

    cout << fixed << setprecision(6);
    cout << "--- CUDA Kernel Launch Latency ---" << endl;
    cout << "Iterations: " << TEST_ITERATIONS << endl;
    cout << "Average Latency: " << avg << " ms (" << avg * 1000.0f << " us)" << endl;
    cout << "Std Deviation:   " << stdev << " ms" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}