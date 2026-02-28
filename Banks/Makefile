all:
	nvcc --resource-usage banks.cu -o banks.x
	./banks.x
	nvcc --resource-usage banks_shared.cu -o banks_shared.x
	./banks_shared.x

multi_kernel:
	nvcc --resource-usage banks_multi_kernel.cu -o banks_multi_kernel.x
	./banks_multi_kernel.x

multi_sm:
	nvcc --resource-usage banks_multi_SM.cu -o banks_multi_SM.x
	./banks_multi_SM.x


clean:
	rm -f *~ *.x
