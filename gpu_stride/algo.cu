#include <cstdio>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"

#define SEED 42
#define gpuErrorCheck(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
#define USE_KMEAN_PLUS
// #define PG_BAR

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);

    if (abort) {
      exit(code);
    }
  }
}

#ifdef LESSMEMORY
__global__ void kmeansKernel(const float *data, float *centroids,
                             int *assigned_cluster, const int numData,
                             const int numFeatures, const int cluster) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float diff, distance;
  for (int idx = tid; idx < numData; idx += stride) {
    float minDistance = 1E38;
    int minCluster = -1;
    for (int i = 0; i < cluster; ++i) {
      distance = 0.0f;
      for (int j = 0; j < numFeatures; ++j) {
        diff = data[idx * numFeatures + j] - centroids[i * numFeatures + j];
        distance += diff * diff;
      }
      if (distance < minDistance) {
        minDistance = distance;
        minCluster = i;
      }
    }

    assigned_cluster[idx] = minCluster;
  }
}

__global__ void updateCentroidsKernel(const float *data, float *centroids,
                                      const int *assigned_cluster,
                                      const int numData, const int numFeatures,
                                      const int cluster) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int idx = tid; idx < cluster; idx += stride) {
    int clusterSize = 0;
    float *centroid_sum =
        (float *)malloc(sizeof(float) * numFeatures * cluster);
    memset(centroid_sum, 0, sizeof(float) * numFeatures * cluster);

    for (int i = 0; i < numData; ++i) {
      if (assigned_cluster[i] == idx) {
        for (int j = 0; j < numFeatures; ++j) {
          centroid_sum[idx * numFeatures + j] += data[i * numFeatures + j];
        }
        clusterSize++;
      }
    }

    if (clusterSize > 0) {
      for (int i = 0; i < numFeatures; ++i) {
        centroids[idx * numFeatures + i] =
            centroid_sum[idx * numFeatures + i] / clusterSize;
      }
    }

    free(centroid_sum);
  }
}
#else
__global__ void kmeansKernel(const float *data, float *centroids,
                             float *new_centroids, int *count_centroids,
                             int *assigned_cluster, const int numData,
                             const int numFeatures, const int cluster) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  float diff, distance;
  for (int idx = tid; idx < numData; idx += stride) {
    float minDistance = 1E38;
    int minCluster = -1;
    for (int i = 0; i < cluster; ++i) {
      distance = 0.0f;
      for (int j = 0; j < numFeatures; ++j) {
        diff = data[idx * numFeatures + j] - centroids[i * numFeatures + j];
        distance += diff * diff;
      }
      if (distance < minDistance) {
        minDistance = distance;
        minCluster = i;
      }
    }

    assigned_cluster[idx] = minCluster;
    atomicAdd(&count_centroids[minCluster], 1);
    for (int i = 0; i < numFeatures; ++i) {
      atomicAdd(&new_centroids[minCluster * numFeatures + i],
                data[idx * numFeatures + i]);
    }
  }
}

__global__ void updateCentroidsKernel(const float *data, float *centroids,
                                      float *new_centroids,
                                      int *count_centroids,
                                      const int *assigned_cluster,
                                      const int numData, const int numFeatures,
                                      const int cluster) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int idx = tid; idx < cluster; idx += stride) {
    int clusterSize = count_centroids[idx];
    if (clusterSize > 0) {
      for (int i = 0; i < numFeatures; ++i) {
        centroids[idx * numFeatures + i] =
            new_centroids[idx * numFeatures + i] / clusterSize;
        new_centroids[idx * numFeatures + i] = 0.0f;
      }
      count_centroids[idx] = 0;
    }
  }
}
#endif

template <typename T>
inline void assignData(std::vector<T> &destination, const int start_des,
                       const std::vector<T> &source, const int start_source,
                       const int numFeatures) {
  for (int k = 0; k < numFeatures; ++k) {
    destination.at(start_des * numFeatures + k) =
        source.at(start_source * numFeatures + k);
  }
}

IntFloatVecFloat kmeans(const std::vector<float> &data, const int numFeatures,
                        const int cluster, const int maxIterations) {
  int numData = data.size() / numFeatures;
  float *d_data;
  float *d_centroids;
  int *d_assignedCluster;

#ifndef LESSMEMORY
  float *d_new_centroids;
  int *d_count_centroids;
#endif

  int kmeanBlockSize = 128;
  int kmeanNumBlocks = 256;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  gpuErrorCheck(
      cudaMalloc((void **)&d_data, sizeof(float) * numFeatures * numData));
  gpuErrorCheck(
      cudaMalloc((void **)&d_centroids, sizeof(float) * numFeatures * cluster));
  gpuErrorCheck(cudaMalloc((void **)&d_assignedCluster, sizeof(int) * numData));
  gpuErrorCheck(cudaMemcpy(d_data, data.data(),
                           sizeof(float) * numFeatures * numData,
                           cudaMemcpyHostToDevice));
#ifndef LESSMEMORY
  gpuErrorCheck(cudaMalloc((void **)&d_new_centroids,
                           sizeof(float) * numFeatures * cluster));
  gpuErrorCheck(cudaMalloc((void **)&d_count_centroids, sizeof(int) * cluster));
  gpuErrorCheck(
      cudaMemset(d_new_centroids, 0.0f, sizeof(float) * numFeatures * cluster));
  gpuErrorCheck(cudaMemset(d_count_centroids, 0, sizeof(int) * cluster));
#endif

  std::vector<float> centroids(numFeatures * cluster);
  std::random_device rd;
  std::mt19937 generator(SEED);
  std::uniform_int_distribution<int> distribution(0, numData - 1);

#ifdef USE_KMEAN_PLUS
  // Assign points with kmeans++
  int firstCentroidIndex = distribution(generator);
  assignData(centroids, 0, data, firstCentroidIndex, numFeatures);

  std::vector<float> distances(numData);
  for (int i = 1; i < cluster; ++i) {
    std::fill(distances.begin(), distances.end(),
              std::numeric_limits<float>::max());
    float totalDistances = 0.0;

    // Compute distances from each data point to the closest centroid
    for (int entry = 0; entry < numData; ++entry) {
      for (int j = 0; j < i; ++j) {
        float distance = 0.0f;
        for (int k = 0; k < numFeatures; ++k) {
          float diff =
              data[entry * numFeatures + k] - centroids[j * numFeatures + k];
          distance += diff * diff;
        }
        distances[entry] = std::min(distances[entry], distance);
      }
      totalDistances += distances[entry];
    }

    // Choose the next centroid based on distances
    std::partial_sum(distances.begin(), distances.end(), distances.begin());
    std::uniform_real_distribution<float> randomProbability(0.0,
                                                            totalDistances);
    float targetProb = randomProbability(generator);

    int selectedDataPointIndex =
        std::lower_bound(distances.begin(), distances.end(), targetProb) -
        distances.begin();
    assignData(centroids, i, data, selectedDataPointIndex, numFeatures);
  }
#else
  std::set<int> centroid_set;
  for (int i = 0; i < cluster; ++i) {
    int new_centroid = distribution(generator);
    while (centroid_set.find(new_centroid) != centroid_set.end()) {
      new_centroid = distribution(generator);
    }
    centroid_set.insert(new_centroid);
    assignData(centroids, i, data, new_centroid, numFeatures);
  }
#endif
  printf("Finish intialization\n");

  gpuErrorCheck(cudaMemcpy(d_centroids, centroids.data(),
                           sizeof(float) * numFeatures * cluster,
                           cudaMemcpyHostToDevice));
#ifdef LESSMEMORY
  gpuErrorCheck(
      cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024));
#endif
  gpuErrorCheck(cudaEventRecord(start, 0));
  for (int iter = 0; iter < maxIterations; ++iter) {
#ifdef PG_BAR
    custom::printProgress((double)(iter + 1) / maxIterations);
#endif

#ifdef LESSMEMORY
    kmeansKernel<<<kmeanNumBlocks, kmeanBlockSize>>>(
        d_data, d_centroids, d_assignedCluster, numData, numFeatures, cluster);
    gpuErrorCheck(cudaDeviceSynchronize());

    updateCentroidsKernel<<<kmeanNumBlocks, kmeanBlockSize>>>(
        d_data, d_centroids, d_assignedCluster, numData, numFeatures, cluster);
    gpuErrorCheck(cudaDeviceSynchronize());
#else
    kmeansKernel<<<kmeanNumBlocks, kmeanBlockSize>>>(
        d_data, d_centroids, d_new_centroids, d_count_centroids,
        d_assignedCluster, numData, numFeatures, cluster);
    gpuErrorCheck(cudaDeviceSynchronize());

    updateCentroidsKernel<<<kmeanNumBlocks, kmeanBlockSize>>>(
        d_data, d_centroids, d_new_centroids, d_count_centroids,
        d_assignedCluster, numData, numFeatures, cluster);
    gpuErrorCheck(cudaDeviceSynchronize());
#endif
  }
#ifdef PG_BAR
  printf("\n");
#endif
  gpuErrorCheck(cudaEventRecord(stop, 0));
  gpuErrorCheck(cudaEventSynchronize(stop));

  float elapsedTime;
  gpuErrorCheck(cudaEventElapsedTime(&elapsedTime, start, stop));

  std::vector<int> h_assignedCluster(numData);
  gpuErrorCheck(cudaMemcpy(centroids.data(), d_centroids,
                           sizeof(float) * numFeatures * cluster,
                           cudaMemcpyDeviceToHost));
  gpuErrorCheck(cudaMemcpy(h_assignedCluster.data(), d_assignedCluster,
                           sizeof(int) * numData, cudaMemcpyDeviceToHost));

  gpuErrorCheck(cudaFree(d_data));
  gpuErrorCheck(cudaFree(d_centroids));
  gpuErrorCheck(cudaFree(d_assignedCluster));

  return std::make_tuple(h_assignedCluster, centroids, elapsedTime);
}
