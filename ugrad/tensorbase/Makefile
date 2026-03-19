all:
	nvcc -std=c++20 -arch=sm_61 --compiler-options '-fPIC -fopenmp' -shared -o libtensor.so -O2 tensor.cpp cpu.cpp -x cu cuda.cu
cpu:
	g++ -std=c++20 -DCPU_ONLY -fPIC -fopenmp -shared -o libtensor.so -O2 tensor.cpp cpu.cpp

clean:
	rm *.so
