#include "data.h"
#include <vector>
#include <math.h>
#include <iostream>

using namespace std;

vector<data> split(vector<data> d) {
  vector<data> res;

  for (int i = 0; i < d.size(); i++) {
    vector<float> img1;
    vector<float> img2;
    vector<float> img3;
    vector<float> img4;

    for (int j = 0; j < d[i].features.size(); j++) {

      if (j<d[i].features.size()/2) {
        if ((j%96)<48) {
          img1.push_back(d[i].features[j]);
        }else{
          img2.push_back(d[i].features[j]);
        }
      }else{
        if ((j%96)<48) {
          img3.push_back(d[i].features[j]);
        }else{
          img4.push_back(d[i].features[j]);
        }
      }
    }

    data curr1;
    data curr2;
    data curr3;
    data curr4;
    curr1.label = d[i].label;
    curr2.label = d[i].label;
    curr3.label = d[i].label;
    curr4.label = d[i].label;

    curr1.features = img1;
    res.push_back(curr1);

    curr2.features = img2;
    res.push_back(curr2);

    curr3.features = img3;
    res.push_back(curr3);

    curr4.features = img4;
    res.push_back(curr4);

  }

  return res;
}

vector<data> split(vector<data> d, int nbPatch) {
  vector<data> res;
  int sqrtPatch = sqrt(nbPatch);

  for (int i = 0; i < d.size(); i++) {
    vector<data> tmp(nbPatch);
    for (int j = 0; j < nbPatch; j++) {
      tmp[j].label = d[i].label;
    }
    int sqrtData = sqrt(d[i].features.size() / 3);
    for (int j = 0; j < sqrtData; j++) {
      for (int k = 0; k < sqrtData; k++) {
        int idx1 = j / (sqrtData / sqrtPatch);
        int idx2 = k / (sqrtData / sqrtPatch);
        tmp[idx1 * sqrtPatch + idx2].features.push_back(d[i].features[j * sqrtData * 3 + k * 3]);
      }
    }
    res.insert(res.end(), tmp.begin(), tmp.end());
  }
  return res;
}
