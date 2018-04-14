#include "k_means.h"
#include "../utils/string_utils.h"
#include "../utils/res.h"
#include "../utils/images.h"
#include <limits>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>



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
    cout << '\r' << 100 * (int) (i + 1.) / nb_iter << "%" << flush;
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

/**
  * Test - K_Means_2
  */
K_Means_2::K_Means_2(string fileName) {
  ifstream in(fileName);
	if (!in) {
		cout << "Can't open file !" << endl;
		exit(0);
	}
	string line;
	getline(in, line);
  vector<string> splittedLine = split(line, ' ');
  this->nbCenters = stoi(splittedLine[0]);
  this->nbFeatures = stoi(splittedLine[1]);
	this->centers = vector<vector<float>>();
	while (getline(in, line)) {
		splittedLine = split(line, ' ');
		vector<float> cluster;
		for (int i = 0; i < splittedLine.size(); i++) {
			cluster.push_back(stof(splittedLine[i]));
		}
		this->centers.push_back(cluster);
	}
  in.close();
}

K_Means_2::K_Means_2(int nbCenters, int nbFeatures) {
  this->nbCenters = nbCenters;
  this->nbFeatures = nbFeatures;
  this->centers = vector<vector<float>>();
  for (int i = 0; i < this->nbCenters; i++) {
    vector<float> tmp;
    for (int j = 0; j < this->nbFeatures; j++) {
      float r = (float) rand() / RAND_MAX;
      tmp.push_back(r);
    }
    this->centers.push_back(tmp);
  }
}

K_Means_2::K_Means_2(int nbCenters, int nbFeatures, vector<data> sample) {
  this->nbCenters = nbCenters;
  this->nbFeatures = nbFeatures;
  float lastDist = 0.f;
  for (int i = 0; i < this->nbCenters; i++) {
    int r = sample.size() * (float) rand() / RAND_MAX;
    this->centers.push_back(vector<float>(sample[r].features));
  }
}

float distance(vector<float> p1, vector<float> p2) {
  float res = 0.f;
  for (int i = 0; i < p1.size(); i++) {
    res += pow(p1[i] - p2[i], 2.);
  }
  return sqrt(res);
}

void K_Means_2::update(vector<data> d) {
  vector<int> assoc;
  vector<int> count(this->nbCenters, 0);
  for (int i = 0; i < d.size(); i++) {
    int pred = predict(d[i]);
    assoc.push_back(pred);
    count[pred]++;
  }
  vector<vector<float>> new_centers;
  for (int k = 0; k < this->nbCenters; k++) {
    new_centers.push_back(vector<float>(this->nbFeatures, 0.f));
  }
  for (int i = 0; i < d.size(); i++) {
    int k = assoc[i];
    for (int j = 0; j < this->nbFeatures; j++) {
      float tmp = count[k] != 0 ? d[i].features[j] / count[k] : 0.f;
      new_centers[k][j] += tmp;
    }
  }
  assoc.clear();
  count.clear();
  this->centers = new_centers;
  new_centers.clear();
}

int K_Means_2::predict(data d) {
  float min = numeric_limits<float>::max();
  int idx = -1;
  for (int k = 0; k < this->nbCenters; k++) {
    float dist = distance(this->centers[k], d.features);
    if (dist < min) {
      idx = k;
      min = dist;
    }
  }
  return idx;
}

vector<float> K_Means_2::predictVec(data d) {
  vector<float> res;
  float moy = 0.f;
  for (int k = 0; k < this->nbCenters; k++) {
    float zk = distance(d.features, this->centers[k]);
    res.push_back(zk);
    moy += zk;
  }
  moy /= this->nbCenters;
  for (int k = 0; k < this->nbCenters; k++) {
    res[k] = max(0.f, moy - res[k]);
  }
  return res;
}



float K_Means_2::score(vector<data> testData) {
  int nbErr = 0;
  int total = testData.size();
  for (data d : testData) {
    nbErr += d.label != predict(d) ? 1 : 0;
  }
  return (float) nbErr / total;
}

/**
  * Première ligne :
  * [Nombre de Centres] [Nombre de features]
  * Après une ligne par vecteur de centre
  */
void K_Means_2::toFile() {
  char name[50];
  sprintf(name, "./res/K_Means_2_%d_%d.txt", this->nbCenters, this->nbFeatures);
  ofstream outFile(name);
  outFile << this->nbCenters << " " << this->nbFeatures << "\n";
  if (outFile.is_open()) {
    for (int i = 0; i < this->nbCenters; i++) {
      for (int j = 0; j < this->nbFeatures; j++) {
        outFile << fixed << setprecision(8) << this->centers[i][j] << " ";
      }
      outFile << "\n";
    }
    outFile.close();
  }
}

void K_Means_2::makeImageCenters() {
  for (int i = 0; i < this->nbCenters; i++) {
      char name[40];
      sprintf(name, "K_Means_%d_%d_features_%d.ppm", this->nbCenters, this->nbFeatures, i);
      int taille = (int) sqrt(this->centers[i].size() / 3.f);
      writeImg(name, this->centers[i], taille, taille);
  }
}
