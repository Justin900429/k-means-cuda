#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60
typedef std::tuple<std::vector<int>, std::vector<float>, float>
    IntFloatVecFloat;

namespace custom {
void printProgress(double percentage) {
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}
}  // namespace custom

#endif