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
  // vector<data> res;
  // int sqrtPatch = sqrt(nbPatch);
  //
  // for (int i = 0; i < d.size(); i++) {
  //   vector<data> tmp(nbPatch);
  //   for (int j = 0; j < nbPatch; j++) {
  //     tmp[j].label = d[i].label;
  //   }
  //   int sqrtData = sqrt(d[i].features.size());
  //   for (int y = 0; y < sqrtData; y++) {
  //     for (int x = 0; x < sqrtData; x++) {
  //       int idx1 = y / (sqrtData / sqrtPatch);
  //       int idx2 = x / (sqrtData / sqrtPatch);
  //       tmp[idx1 * sqrtPatch + idx2].features.push_back(d[i].features[y * sqrtData + x]);
  //     }
  //   }
  //   res.insert(res.end(), tmp.begin(), tmp.end());
  // }
  vector<data> res;
  for (int i = 0; i < d.size(); i++) {
    vector<data> tmp(nbPatch);
    int n = sqrt(d[i].features.size() / 3);
    n *= 3;
    int sqrtPatch = sqrt(nbPatch);
    for (int idx = 0; idx < d[i].features.size(); idx++) {
      int x = idx % n;
      int y = idx / (n / 3);

      int xPatch = sqrtPatch * (float) x / n;
      int yPatch = sqrtPatch * (float) y / n;
      tmp[yPatch * sqrtPatch + xPatch].features.push_back(d[i].features[idx]);
      tmp[yPatch * sqrtPatch + xPatch].label = d[i].label;
    }
    res.insert(res.begin(), tmp.begin(), tmp.end());
  }
  return res;
}
