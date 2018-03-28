#include "k_means.h"
#include <limits>
#include <math.h>
#include <iostream>



K_means::K_means(int n){
  this->nb_clusters = n;
}

float K_means::EuclidianDistance(float* x, float* y){

  float res = 0;
  // std::cout << sizeof(x)/sizeof(x[0]) << '\n';
  for (int i = 0; i < sizeof(x)/sizeof(*x); i++) {
    res += pow((x[i] - y[i] ),2);
  }
  return sqrt(res);

}
