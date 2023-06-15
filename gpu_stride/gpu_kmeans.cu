#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

typedef std::tuple<std::vector<int>, std::vector<float>, float>
    IntFloatVecFloat;
extern IntFloatVecFloat kmeans(const std::vector<float> &data,
                               const int numFeatures, const int cluster,
                               const int maxIterations);

int main(int argc, char *argv[]) {
  int cluster = 20, maxIterations = 300;

  if (argc >= 2) {
    cluster = atoi(argv[1]);
  }
  if (argc >= 3) {
    cluster = atoi(argv[1]);
    maxIterations = atoi(argv[2]);
  }
  if (argc >= 4) {
    printf(
        "Invalid arguments. Please provide `cluster (default to 20)`, "
        "`maxIterations (default to 300)`\n");
    return 1;
  }

  std::vector<float> data;
  std::vector<int> label;

  std::ifstream file("data.txt");
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream iss(line);

    float temp_x, temp_y;
    int temp_label;
    if (!(iss >> temp_x >> temp_y >> temp_label)) {
      printf("Error reading file.\n");
      continue;
    }

    data.push_back(temp_x);
    data.push_back(temp_y);
    label.push_back(temp_label);
  }
  printf("Finishing reading data.\n");

  IntFloatVecFloat res = kmeans(data, 2, cluster, maxIterations);

  int numData = data.size() / 2;

#ifdef LESSMEMORY
  std::string file_name = std::string("output/gpu_stride_kmeans_ori") + '_' +
#else
  std::string file_name = std::string("output/gpu_stride_kmeans_imp") + '_' +
#endif
                          std::to_string(cluster) + '_' +
                          std::to_string(numData) + ".txt";
  std::ofstream outputFile(file_name);
  if (!outputFile.is_open()) {
    printf("Failed to open the output file.\n");
    return 1;
  }

  outputFile << std::get<2>(res) << "\n";
  outputFile << cluster << " " << numData << "\n";
  for (int idx = 0; idx < cluster; ++idx) {
    outputFile << std::get<1>(res).at(2 * idx) << " "
               << std::get<1>(res).at(2 * idx + 1) << "\n";
  }
  for (int idx = 0; idx < numData; ++idx) {
    outputFile << data.at(2 * idx) << " " << data.at(2 * idx + 1) << " "
               << label.at(idx) << " " << std::get<0>(res).at(idx) << "\n";
  }

  outputFile.close();
}