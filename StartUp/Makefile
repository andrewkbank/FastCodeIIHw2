all:
	nvcc startup.cu -o startup.x --resource-usage
	nvcc emptystartup.cu -o emptystartup.x --resource-usage

compare_allocations:
	nvcc startup.cu -o startup.x --resource-usage
	./startup.x

empty_startup:
	nvcc emptystartup.cu -o emptystartup.x --resource-usage
	./emptystartup.x

profiler:
	nvcc profiler.cu -o profiler.x --resource-usage
	nsys nvprof ./profiler.x

profiler_empty:
	nvcc emptystartup.cu -o emptystartup.x --resource-usage
	nsys nvprof ./emptystartup.x

clean:
	rm -f *.o *~ *.x *.nsys-rep *.sqlite
