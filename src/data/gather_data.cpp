#include "gather_data.h"
#include <iostream>

vector<data> gatherDataFeatures(K_means k_means_patch, vector<data> splittedData, int N) {
  vector<data> gatherDataFeatures;
  for (int i = 0; i < splittedData.size(); i += 4) {
    data fst = splittedData[i];
    data snd = splittedData[i + 1];
    data thr = splittedData[i + 2];
    data fth = splittedData[i + 3];
    int fst_pred = k_means_patch.predict(fst);
    int snd_pred = k_means_patch.predict(snd);
    int thr_pred = k_means_patch.predict(thr);
    int fth_pred = k_means_patch.predict(fth);
    data res;
    res.features = vector<float>();
    for (int j = 0; j < N; j++) {
        res.features.push_back(fst_pred == i ? 1.f : 0.f);
    }
    for (int j = 0; j < N; j++) {
        res.features.push_back(snd_pred == i ? 1.f : 0.f);
    }
    for (int j = 0; j < N; j++) {
        res.features.push_back(thr_pred == i ? 1.f : 0.f);
    }
    for (int j = 0; j < N; j++) {
        res.features.push_back(fth_pred == i ? 1.f : 0.f);
    }
    if (fst.label != snd.label && snd.label != thr.label && thr.label != fth.label) {
      cout << "Not same label of 4 consecutive pathes !" << endl;
      exit(1);
    }
    res.label = fst.label;
    gatherDataFeatures.push_back(res);
  }
  return gatherDataFeatures;
}

pair<vector<data>, K_means > trainKMeansDataFeatures(vector<data> splittedData, int N, int nbIter) {
  K_means k_means(N, splittedData);
  k_means.proc(nbIter);

  vector<data> newData = gatherDataFeatures(k_means, splittedData, N);

  return make_pair(newData, k_means);
}
