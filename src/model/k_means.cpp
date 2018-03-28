#include "k_means.h"
#include <limits>
#include <math.h>
#include <iostream>



K_means::K_means(int n, vector<data> dat){
  this->nb_clusters = n;
  this->dat = dat;
  for (int i = 0; i < n; i++) {
    auto r = (int)( (float) std::rand() / RAND_MAX * dat.size());
    this->center.push_back( vector<float> (dat[r].features, dat[r].features + 3072));

  }
}

float K_means::EuclidianDistance(vector<float> x, vector<float> y){
  float res = 0;
  for (int i = 0; i < 3072; i++) {
    res += pow((x[i] - y[i] ),2);
  }
  return sqrt(res);

}

void K_means::proc(int nb_iter){
  std::vector<int> assoc;
  for (int i = 0; i < nb_iter; i++) {
    assoc.clear();
    std::cout << ((float) i/nb_iter)*100 << "%" << '\n';
    for (int j = 0; j < this->dat.size(); j++) {
      float plusPetit = std::numeric_limits<float>::infinity();
      int indice = 0;
      for (int k = 0; k < this->nb_clusters; k++) {
        float tmp = EuclidianDistance(center[k],vector<float> (dat[j].features, dat[j].features + 3072));
        if (tmp<plusPetit) {
          plusPetit = tmp;
          indice = k;
        }
      }
      assoc.push_back(indice);

    }
    std::vector<int> nb_assoc (this->nb_clusters,0);
    std::vector<std::vector<float> > new_centre (this->nb_clusters, std::vector<float> (3072,0) );
    for (size_t i = 0; i < assoc.size(); i++) {
      for (size_t j = 0; j < 3072; j++) {
        new_centre[assoc[i]][j] += this->dat[assoc[i]].features[j];
        nb_assoc[assoc[i]] ++;
      }
    }
    for (size_t i = 0; i < this->nb_clusters; i++) {
      for (size_t j = 0; j < 3072; j++) {
        new_centre[i][j] /= nb_assoc[i];
      }
    }
    this->center = new_centre;
  }

  for (size_t i = 0; i < assoc.size(); i++) {
    std::cout << i << ":" << assoc[i] << '\n';
  }
}
