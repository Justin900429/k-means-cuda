CC=g++
LIBDIRS=-L/usr/local/cuda-11.7/lib64
INCLUDES=-I/usr/local/cuda-11.7/include
LINKER_FLAGS=-lcudart -lcuda
NVCC=nvcc

all: cpu_kmeans gpu_block_kmeans gpu_stride_kmeans_imp gpu_stride_kmeans_ori

compare: cpu_kmeans gpu_stride_kmeans_imp gpu_stride_kmeans_ori

cpu_kmeans: cpu_kmeans.o
	$(CC) -o cpu_kmeans cpu/cpu_kmeans.o

cpu_kmeans.o: cpu/cpu_kmeans.cpp
	$(CC) -c cpu/cpu_kmeans.cpp -o cpu/cpu_kmeans.o

gpu_block_kmeans: gpu_block_kmeans.o gpu_block_algo.o
	$(CC) $(LIBDIRS) gpu_block/gpu_kmeans.o gpu_block/algo.o -o gpu_block_kmeans $(LINKER_FLAGS)

gpu_block_kmeans.o: gpu_block/gpu_kmeans.cu
	$(NVCC) -c gpu_block/gpu_kmeans.cu -o gpu_block/gpu_kmeans.o

gpu_block_algo.o: gpu_block/algo.cu
	$(NVCC) -c gpu_block/algo.cu -o gpu_block/algo.o

gpu_stride_kmeans_imp: gpu_stride_kmeans_imp.o gpu_stride_algo_imp.o
	$(CC) $(LIBDIRS) gpu_stride/gpu_kmeans_imp.o gpu_stride/algo_imp.o -o gpu_stride_kmeans_imp $(LINKER_FLAGS)

gpu_stride_kmeans_ori: gpu_stride_kmeans_ori.o gpu_stride_algo_ori.o
	$(CC) $(LIBDIRS) gpu_stride/gpu_kmeans_ori.o gpu_stride/algo_ori.o -o gpu_stride_kmeans_ori $(LINKER_FLAGS)

gpu_stride_kmeans_imp.o: gpu_stride/gpu_kmeans.cu
	$(NVCC) -c gpu_stride/gpu_kmeans.cu -o gpu_stride/gpu_kmeans_imp.o

gpu_stride_kmeans_ori.o: gpu_stride/gpu_kmeans.cu
	$(NVCC) -DLESSMEMORY -c gpu_stride/gpu_kmeans.cu -o gpu_stride/gpu_kmeans_ori.o

gpu_stride_algo_imp.o: gpu_stride/algo.cu
	$(NVCC) -c gpu_stride/algo.cu -o gpu_stride/algo_imp.o

gpu_stride_algo_ori.o: gpu_stride/algo.cu
	$(NVCC) -DLESSMEMORY -c gpu_stride/algo.cu -o gpu_stride/algo_ori.o

clean:
	rm -f gpu_block/*.o gpu_stride/*.o cpu/*.o gpu_stride_kmeans* *_dbscan