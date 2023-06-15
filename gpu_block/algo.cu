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

__global__ void kmeansKernel(const float *data, float *centroids,
                             int *assigned_cluster, const int numData,
                             const int numFeatures, const int cluster) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numData) {
    return;
  }

  float distance, diff, minDistance = 1E38;
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

__global__ void updateCentroidsKernel(const float *data, float *centroids,
                                      const int *assigned_cluster,
                                      const int numData, const int numFeatures,
                                      const int cluster) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= cluster) {
    return;
  }

  int clusterSize = 0;
  float *centroid_sum = (float *)malloc(sizeof(float) * numFeatures * cluster);
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

  int kmeanBlockSize = 128;
  int kmeanNumBlocks = (numData + kmeanBlockSize - 1) / kmeanBlockSize;

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

  std::vector<float> centroids(numFeatures * cluster);
  std::random_device rd;
  std::mt19937 generator(SEED);
  std::uniform_int_distribution<int> distribution(0, numData - 1);

#ifdef USE_KMEAN_PLUS
  // Assign points with kmeans++
  for (int i = 0; i < cluster; ++i) {
    std::vector<float> distances(numData, std::numeric_limits<float>::max());
    float totalDistances = 0.0;
    for (int entry = 0; entry < numData; ++entry) {
      for (int j = 0; j < i; ++j) {
        float diff, distance = 0.0f;
        for (int k = 0; k < numFeatures; ++k) {
          diff = data.at(entry * numFeatures + k) -
                 centroids.at(j * numFeatures + k);
          distance += diff * diff;
        }
        distances[entry] = std::min(distances[entry], distance);
      }
      totalDistances += distances[entry];
    }

    std::discrete_distribution<int> discreteDistribution(distances.begin(),
                                                         distances.end());
    float cumulativeProb = 0.0;
    float targetProb =
        std::uniform_real_distribution<float>(0.0, totalDistances)(generator);

    for (int j = 0; j < numData; ++j) {
      cumulativeProb += distances[j];
      if (cumulativeProb >= targetProb) {
        assignData(centroids, i, data, j, numFeatures);
        break;
      }
    }
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

  gpuErrorCheck(cudaMemcpy(d_centroids, centroids.data(),
                           sizeof(float) * numFeatures * cluster,
                           cudaMemcpyHostToDevice));
  gpuErrorCheck(
      cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024));
  gpuErrorCheck(cudaEventRecord(start, 0));
  for (int iter = 0; iter < maxIterations; ++iter) {
#ifdef PG_BAR
    custom::printProgress((double)(iter + 1) / maxIterations);
#endif
    kmeansKernel<<<kmeanNumBlocks, kmeanBlockSize>>>(
        d_data, d_centroids, d_assignedCluster, numData, numFeatures, cluster);
    gpuErrorCheck(cudaDeviceSynchronize());

    updateCentroidsKernel<<<kmeanNumBlocks, kmeanBlockSize>>>(
        d_data, d_centroids, d_assignedCluster, numData, numFeatures, cluster);
    gpuErrorCheck(cudaDeviceSynchronize());
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
