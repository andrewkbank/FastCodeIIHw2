#include <iostream>

using namespace std;
/**
 * The original kernel_call is kept unmodified to verify my implementation
 *
 * kernel_call_andrew is entirely written by Gemini 3.0 flash. I gave it the
 * thread number, warp dimensions, and the thread block dimensions I came up
 * with in the hw, and let it generate the proper code based on the original
 * kernel_call.
 *
 * The dimensions were chosen without considering bank conflicts, and assuming
 * that a single thread block can have 1024 threads maximum. Ideally, we would
 * want more threads (since there are 4096 outputs) and ideally, we would change
 * the shared memory organization to mitigate bank conflicts, but that is
 * outside the scope of this part of the hw.
 *
 * The main method was modified by Gemini to include a checksum to validate
 * kernel_call_andrew, and it calculates TFLOPS performance for us
 *
 * Overall, kernel_call_andrew's performance isn't great (~0.01 TFLOPs vs 8.2
 * theoretical maximum TFLOPs) This is likely due to the fact that we are
 * restricted to one thread block, and that there are bank conflicts galore.
 */
__global__ void kernel_call(float *c) {

    __shared__ float buffer[12 * 1024];

    float *s_c = buffer;
    float *s_a = buffer + 4096;
    float *s_b = buffer + 4096 * 2;

    // 1 threadblock only
    int id = threadIdx.x;
    int p = blockDim.x;

    /**** Do Not Change Code Above This ****/

    for (int i = id; i < 64; i += p) {
        for (int j = id; j < 64; j += p) {
            s_c[i * 64 + j] = i * 64 + j + 1.0;
            s_a[i * 64 + j] = i * 64 + j + 2.0;
            s_b[i * 64 + j] = i * 64 + j + 1.0;
        }
    }

    // ensure all threads are done initializing the buffer
    __syncthreads();

    // Computes C += A * B using only 1 thread
    // A is column major order, the other 2 matrices are row major order
    for (int i = 0; i < 64; ++i) {         // 64 rows of C
        for (int j = 0; j < 64; ++j) {     // 64 columns of C
            for (int p = 0; p < 64; ++p) { // 64 columns of A
                s_c[i * 64 + j] += s_a[p * 64 + i] * s_b[p * 64 + j];
            }
        }
    }

    /**** Do Not Change Code Below This ****/

    // copy C out such that C is in row major order
    for (int i = id; i < 64 * 64; i += p) {
        c[i] = s_c[i];
    }
}

__global__ void kernel_call_andrew(float *c) {

    __shared__ float buffer[12 * 1024];

    float *s_c = buffer;
    float *s_a = buffer + 4096;
    float *s_b = buffer + 4096 * 2;

    // 1 threadblock only
    int id = threadIdx.x;
    int p = blockDim.x;

    /**** Do Not Change Code Above This ****/

    // 1. Initialization: Use all 1024 threads to fill shared memory fast
    // Each thread handles 4 elements (4096 / 1024)
    for (int idx = id; idx < 4096; idx += p) {
        s_c[idx] = idx + 1.0f;
        s_a[idx] = idx + 2.0f;
        s_b[idx] = idx + 1.0f;
    }

    __syncthreads();

    // 2. Map 1D thread ID to 4x8 Warp Tile coordinates
    int warp_id = id / 32;
    int lane_id = id % 32;

    // Position of this thread's 4x8 tile relative to the 32x32 block area
    int tile_base_i = (warp_id / 4) * 4 + (lane_id / 8);
    int tile_base_j = (warp_id % 4) * 8 + (lane_id % 8);

    // 3. Compute C += A * B
    // We cover the 64x64 matrix by shifting our 32x32 "block footprint"
    // in 4 steps (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
    for (int block_row_off = 0; block_row_off < 64; block_row_off += 32) {
        for (int block_col_off = 0; block_col_off < 64; block_col_off += 32) {

            int i = tile_base_i + block_row_off;
            int j = tile_base_j + block_col_off;

            float sum = 0.0f;
            for (int k = 0; k < 64; ++k) {
                // A is column major: s_a[row=k, col=i] -> s_a[k * 64 + i]
                // B is row major:    s_b[row=k, col=j] -> s_b[k * 64 + j]
                sum += s_a[k * 64 + i] * s_b[k * 64 + j];
            }
            s_c[i * 64 + j] += sum;
        }
    }

    __syncthreads();

    /**** Do Not Change Code Below This ****/

    // copy C out such that C is in row major order
    for (int i = id; i < 64 * 64; i += p) {
        c[i] = s_c[i];
    }
}

float compute_checksum(float *data, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return (float)sum;
}

int main() {
    const int N = 64 * 64;
    float *h_out_orig = (float *)malloc(N * sizeof(float));
    float *h_out_andrew = (float *)malloc(N * sizeof(float));
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_orig, ms_andrew;

    // --- Test Original (1 Thread) ---
    cudaEventRecord(start);
    kernel_call<<<1, 1>>>(d_out);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms_orig, start, stop);
    cudaMemcpy(h_out_orig, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Test Andrew (1024 Threads) ---
    cudaEventRecord(start);
    kernel_call_andrew<<<1, 1024>>>(d_out);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms_andrew, start, stop);
    cudaMemcpy(h_out_andrew, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Verification ---
    float check_orig = compute_checksum(h_out_orig, N);
    float check_andrew = compute_checksum(h_out_andrew, N);

    cout << "Original Kernel (1 thread):    " << ms_orig
         << " ms | Checksum: " << check_orig << endl;
    cout << "Andrew Kernel (1024 threads):  " << ms_andrew
         << " ms | Checksum: " << check_andrew << endl;

    if (abs(check_orig - check_andrew) < 1e-2) {
        cout << "VERIFICATION SUCCESSFUL!" << endl;
    } else {
        cout << "VERIFICATION FAILED!" << endl;
    }
    double total_flops = 2.0 * pow(64, 3);
    double tflops_original = (total_flops) / (ms_orig * 1e9);
    double tflops_andrew = (total_flops) / (ms_andrew * 1e9);

    cout << "1 Thread Performance: " << tflops_original << " TFLOPS" << endl;
    cout << "1024 Thread Performance: " << tflops_andrew << " TFLOPS" << endl;

    free(h_out_orig);
    free(h_out_andrew);
    cudaFree(d_out);
    return 0;
}
