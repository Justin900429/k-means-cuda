#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdio>
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

namespace custom {
void printProgress(double percentage) {
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}

inline double euclidean_distance(double x1, double y1, double x2, double y2) {
  return pow(x1 - x2, 2) + pow(y1 - y2, 2);
}
}  // namespace custom

#endif