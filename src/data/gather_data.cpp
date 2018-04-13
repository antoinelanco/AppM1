#include "gather_data.h"
#include <iostream>

vector<data> gatherDataFeatures(K_Means_2 k_means_patch, vector<data> splittedData, int N, int nbPatch) {
  vector<data> dataFeatures;
  vector<int> count(N, 0);
  for (int i = 0; i < splittedData.size(); i += nbPatch) {
    int label = splittedData[i].label;
    data res;
    res.label = label;
    res.features = vector<float>(N * nbPatch, 0.f);
    for (int j = 0; j < nbPatch; j++) {
      data curr = splittedData[i + j];
      if (curr.label != label) {
        cout << "Not same label of " << nbPatch << " consecutive pathes !" << endl;
        exit(1);
      }

      int pred = k_means_patch.predict(curr);
      count[pred]++;
      res.features[j * N + pred] = 1.f;
      cout << "\r" << 100 * (int) (i + j + 1.f) / splittedData.size() << "%" << flush;
    }
    dataFeatures.push_back(res);
  }
  cout << endl;
  return dataFeatures;
}
