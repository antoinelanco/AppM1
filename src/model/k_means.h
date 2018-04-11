#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>
#include <string>
#include "../data/data.h"

using namespace std;

class K_means {
private:
  vector<data> dat;
  int nb_clusters;
  int nbFeatures;
  vector<vector<float> > center;

public:
  K_means (int n, vector<data> dat);
  void Init();
  float EuclidianDistance(vector<float> x, vector<float> y);
  void proc(int nb_iter);
  int predict(data img);
  float loss(vector<data> test_data);
  //virtual ~K_means ();
};

class K_Means_2 {
private:
  int nbCenters;
  int nbFeatures;
  vector<vector<float>> centers;

public:
  K_Means_2(string fileName);
  K_Means_2(int nbCenters, int nbFeatures);
  K_Means_2(int nbCenters, int nbFeatures, vector<data> sample);
  void update(vector<data> d);
  int predict(data d);
	float score(vector<data> d);
  void toFile();
  void makeImageCenters();
};

#endif
