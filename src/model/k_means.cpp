#include "k_means.h"
#include <limits>
#include <math.h>
#include <iostream>



K_means::K_means(int n){
  this->nb_clusters = n;
}

float K_means::EuclidianDistance(int x1, int x2, int x3, int y1, int y2, int y3){
  return sqrt(pow((x1-y1),2) + pow((x2-y2),2) + pow((x3-y3),2));
}
