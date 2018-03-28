#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>
#include "../utils/data.h"

using namespace std;

class K_means {
private:
  vector<data> dat;
  int nb_clusters;

public:
  K_means (int n);
  float EuclidianDistance(float* x, float* y);
  //virtual ~K_means ();
};




#endif
