#include "k_means.h"
#include <limits>
#include <math.h>
#include <iostream>



K_means::K_means(int n, vector<data> dat){
  this->nb_clusters = n;
  this->dat = dat;
  this->nbFeatures = dat[0].features.size();
  for (int i = 0; i < n; i++) {
    auto r = (int)( (float) rand() / RAND_MAX * dat.size());
    this->center.push_back( vector<float> (dat[r].features));
  }
}

float K_means::EuclidianDistance(vector<float> x, vector<float> y){
  float res = 0;
  for (int i = 0; i < this->nbFeatures; i++) {
    res += pow((x[i] - y[i] ),2);
  }
  return sqrt(res);

}

void K_means::proc(int nb_iter){
  for (int i = 0; i < nb_iter; i++) {
    vector<int> assoc;
    for (int j = 0; j < this->dat.size(); j++) {
      float plusPetit = numeric_limits<float>::infinity();
      int indice = 0;
      for (int k = 0; k < this->nb_clusters; k++) {
        float tmp = EuclidianDistance(center[k], dat[j].features);
        if (tmp<plusPetit) {
          plusPetit = tmp;
          indice = k;
        }
      }
      assoc.push_back(indice);
    }
    vector<int> nb_assoc (this->nb_clusters,0);
    vector<vector<float>> new_centre;
    for (int k = 0; k < this->nb_clusters; k++) {
      new_centre.push_back(vector<float> (this->nbFeatures,0));
    }
    for (size_t k = 0; k < assoc.size(); k++) {
      for (size_t j = 0; j < this->nbFeatures; j++) {
        new_centre[assoc[k]][j] += this->dat[k].features[j];
        nb_assoc[assoc[k]] ++;
      }
    }
    for (size_t k = 0; k < this->nb_clusters; k++) {
      for (size_t j = 0; j < this->nbFeatures; j++) {
        new_centre[k][j] /= nb_assoc[k];
      }
    }
    this->center = new_centre;
    cout << '\r' << ((i+1.)/nb_iter)*100 << "%" << flush;
  }
  cout << '\n';
}

int K_means::predict(data img){
  float plusPetit = numeric_limits<float>::infinity();
  int indice = 0;
  for (int k = 0; k < this->nb_clusters; k++) {
    float tmp = EuclidianDistance(center[k], img.features);
    if (tmp<plusPetit) {
      plusPetit = tmp;
      indice = k;
    }
  }
  return indice;
}

float K_means::loss(vector<data> test_data){
  float res = 0.;
  for (size_t i = 0; i < test_data.size(); i++) {
    if (test_data[i].label != predict(test_data[i])) {
      res+=1;
    }
  }
  return res/test_data.size();
}
