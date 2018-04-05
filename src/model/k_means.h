#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>
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




#endif
