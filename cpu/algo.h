#ifndef ALGO_H
#define ALGO_H

#include <chrono>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include "utils.h"
#define SEED 42
#define EPSILON 0.001
#define USE_KMEAN_PLUS
#define PG_BAR

typedef std::tuple<std::vector<int>, std::vector<float>, float>
    IntFloatVecFloat;
typedef std::tuple<std::vector<int>, std::vector<float>, std::vector<float>>
    ThreeVec;

namespace dbscan {
std::vector<int> regionQuery(const std::vector<float> &x,
                             const std::vector<float> &y, const int cur_id) {
  std::vector<int> neighbors;

  for (int idx = 0; idx < x.size(); ++idx) {
    if (custom::euclidean_distance(x.at(idx), y.at(idx), x.at(cur_id),
                                   y.at(cur_id)) <= EPSILON) {
      neighbors.push_back(idx);
    }
  }

  return neighbors;
}

std::vector<int> DBSCAN(const std::vector<float> &x,
                        const std::vector<float> &y, const int minPts,
                        const float eps) {
  std::vector<int> cluster_label(x.size(), -1);
  std::vector<std::vector<int>> dis(x.size(), std::vector<int>(x.size(), 0));
  int clusterID = 0;

  for (int idx = 0; idx < x.size(); ++idx) {
    for (int idy = idx + 1; idy < x.size(); ++idy) {
      dis[idx][idy] = custom::euclidean_distance(x.at(idx), y.at(idx),
                                                 x.at(idy), y.at(idy));
    }
  }

  for (int idx = 0; idx < x.size(); ++idx) {
    if (cluster_label.at(idx) != -1) {
      continue;
    }

    std::vector<int> neighbors = regionQuery(x, y, idx);

    if (neighbors.size() < minPts) {
      cluster_label.at(idx) = 0;
    } else {
      ++clusterID;
      cluster_label.at(idx) = clusterID;

      for (int idy = 0; idy < neighbors.size(); ++idy) {
        int neighbor = neighbors.at(idy);
        if (cluster_label.at(neighbor) == -1) {
          cluster_label.at(neighbor) = clusterID;

          std::vector<int> neighbor_neighbors = regionQuery(x, y, neighbor);
          if (neighbor_neighbors.size() >= minPts) {
            neighbors.insert(neighbors.end(), neighbor_neighbors.begin(),
                             neighbor_neighbors.end());
          }
        }
      }
    }
  }

  return cluster_label;
}
}  // namespace dbscan

namespace kmeans {
template <typename T>
inline void assignData(std::vector<T> &destination, const int start_des,
                       const std::vector<T> &source, const int start_source,
                       const int numFeatures) {
  for (int k = 0; k < numFeatures; ++k) {
    destination.at(start_des * numFeatures + k) =
        source.at(start_source * numFeatures + k);
  }
}

IntFloatVecFloat KMENAS(const std::vector<float> &data, const int numFeatures,
                        const int cluster, int maxIterations) {
  int numData = data.size() / numFeatures;
  std::random_device rd;
  std::mt19937 generator(SEED);
  std::uniform_int_distribution<int> distribution(0, numData - 1);
  std::vector<float> centroids(numFeatures * cluster);
  std::vector<int> assigned_cluster(numData, 0);

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
    centroids_x.at(i) = x.at(new_centroid);
    centroids_y.at(i) = y.at(new_centroid);
  }
#endif

  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < maxIterations; ++iter) {
#ifdef PG_BAR
    custom::printProgress((float)(iter + 1) / maxIterations);
#endif
    for (int entry = 0; entry < numData; ++entry) {
      float minDistance = std::numeric_limits<float>::max();
      int closestCentroid = 0;

      float distance, diff;
      for (int i = 0; i < cluster; ++i) {
        distance = 0.0f;
        for (int k = 0; k < numFeatures; ++k) {
          diff = data.at(entry * numFeatures + k) -
                 centroids.at(i * numFeatures + k);
          distance += diff * diff;
        }
        if (distance < minDistance) {
          minDistance = distance;
          closestCentroid = i;
        }
      }

      assigned_cluster.at(entry) = closestCentroid;
    }

    std::vector<int> cluster_counts(cluster, 0);
    std::vector<float> new_centroid(numFeatures * cluster, 0.0);

    int cur_cluster;
    for (int entry = 0; entry < numData; ++entry) {
      cur_cluster = assigned_cluster.at(entry);
      cluster_counts.at(cur_cluster) += 1;

      for (int idx = 0; idx < numFeatures; ++idx) {
        new_centroid.at(cur_cluster * numFeatures + idx) +=
            data.at(entry * numFeatures + idx);
      }
    }

    for (int i = 0; i < cluster; ++i) {
      if (cluster_counts.at(i) > 0) {
        for (int j = 0; j < numFeatures; ++j) {
          centroids.at(i * numFeatures + j) =
              new_centroid.at(i * numFeatures + j) / cluster_counts.at(i);
        }
      }
    }
  }
#ifdef PG_BAR
  printf("\n");
#endif
  auto end = std::chrono::high_resolution_clock::now();
  float elapsedTime =
      std::chrono::duration_cast<
          std::chrono::duration<float, std::chrono::milliseconds::period>>(
          end - start)
          .count();
  return std::make_tuple(assigned_cluster, centroids, elapsedTime);
}
}  // namespace kmeans

#endif