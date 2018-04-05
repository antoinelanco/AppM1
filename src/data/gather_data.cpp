#include "gather_data.h"
#include <iostream>

vector<data> gatherDataFeatures(K_means k_means_patch, vector<data> splittedData, int N, int nbPatch) {
  vector<data> gatherDataFeatures;
  for (int i = 0; i < splittedData.size(); i += nbPatch) {
    int label = splittedData[i].label;
    data res;
    res.label = label;
    for (int j = 0; j < nbPatch; j++) {
      data curr = splittedData[i + j];
      if (curr.label != label) {
        cout << "Not same label of " << nbPatch << " consecutive pathes !" << endl;
        exit(1);
      }

      int pred = k_means_patch.predict(curr);
      for (int k = 0; k < N; k++) {
          res.features.push_back(pred == k ? 1.f : 0.f);
      }
    }
    gatherDataFeatures.push_back(res);
  }
  return gatherDataFeatures;
}

pair<vector<data>, K_means > trainKMeansDataFeatures(vector<data> splittedData, int N, int nbIter, int nbPatch) {
  K_means k_means(N, splittedData);
  k_means.proc(nbIter);

  vector<data> newData = gatherDataFeatures(k_means, splittedData, N, nbPatch);

  return make_pair(newData, k_means);
}
