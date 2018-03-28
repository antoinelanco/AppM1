#ifndef K_MEANS_H
#define K_MEANS_H

#include <vector>
#include "../utils/data.h"

using namespace std;

class K_means {
private:
  vector<data> dat;
  int nb_clusters;
  float EuclidianDistance(int x1, int x2, int x3, int y1, int y2, int y3);

public:
  K_means (int n);
  //virtual ~K_means ();
};




#endif
