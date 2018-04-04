#include "data.h"
#include <vector>
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

    data curr;
    curr.label = d[i].label;

    curr.features = img1;
    res.push_back(curr);

    curr.features = img2;
    res.push_back(curr);

    curr.features = img3;
    res.push_back(curr);

    curr.features = img4;
    res.push_back(curr);


    std::cout << img1.size() << '\n';
    std::cout << img2.size() << '\n';
    std::cout << img3.size() << '\n';
    std::cout << img4.size() << '\n';

  }

  return res;
}
