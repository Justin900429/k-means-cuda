#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "algo.h"
#include "utils.h"

// #define DEBUG

int main(int argc, char *argv[]) {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> label;

  std::ifstream file("data.txt");
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream iss(line);

    float temp_x, temp_y;
    int temp_label;
    if (!(iss >> temp_x >> temp_y >> temp_label)) {
      printf("Error reading file.");
      continue;
    }

    x.push_back(temp_x);
    y.push_back(temp_y);
    label.push_back(temp_label);
  }

#ifdef DEBUG
  for (int idx = 0; idx < x.size(); ++idx) {
    printf("%lf %lf %d\n", x.at(idx), y.at(idx), label.at(idx));
  }
#endif

  int minPts = 5;
  float eps = 0.1;

  if (argc >= 2) {
    minPts = atoi(argv[1]);
  }
  if (argc >= 3) {
    eps = atof(argv[2]);
  }
  if (argc >= 4) {
    printf(
        "Invalid arguments. Please provide `cluster (default to 100)`, "
        "`maxIterations (default to 300)`\n");
    return 1;
  }

  std::vector<int> res = dbscan::DBSCAN(x, y, minPts, eps);

  std::ofstream outputFile("output/cpu_dbscan.txt");
  if (!outputFile.is_open()) {
    printf("Failed to open the output file.\n");
    return 1;
  }

  outputFile << 0 << " " << x.size() << "\n";
  for (int idx = 0; idx < x.size(); ++idx) {
    outputFile << x.at(idx) << " " << y.at(idx) << " " << label.at(idx) << " "
               << res.at(idx) << "\n";
  }

  outputFile.close();
}