**How to run my implementations:**

MMM:

make all will compile and run mmm.cu. It will run both the sample version of MMM and my version of MMM and compare their checksum results, execution times, and TFLOPS.

Banks:

make all will compile and run banks.cu and banks_shared.cu. It will compare their runtime and verify both of their correctness

make multi_kernel will compile and run banks_multi_kernel.cu. It will do the transpose on a 4096x4096 matrix and record the runtime + correctness. It launches multiple versions of the kernel from banks_shared.cu except with the proper offsets.

make multi_sm will compile and run banks_multi_SM.cu. It will do the transpose on a 4096x4096 matrix and record the runtime + correctness. It runs the kernel from banks_shared.cu on multiple SMs for the best performance.
