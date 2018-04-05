#include <fstream>
#include <iostream>

#include "read_mnist.h"

vector<data> read_mnist(string dataFileName, string labelFileName, int nbToLoad) {
  vector<data> res;

  ifstream inData(dataFileName.c_str());
  ifstream inLabel(labelFileName.c_str());

  if (!inData || !inLabel) {
      cout << "Error during opening models file" << endl;
      exit(0);
  }
  char unsued[4 * 4];
  inLabel.read(unsued, 2 * 4);
  inData.read(unsued, 4 * 4);

  for (int i = 0; i < nbToLoad; i++) {
    data curr;
    char tmpL = 0;
    inLabel.read(&tmpL, 1);
    curr.label = (unsigned int) tmpL;
    char tmp[28 * 28];
    inData.read(tmp, 28 * 28);
    for (int j = 0; j < 28 * 28; j++) {
      curr.features.push_back((unsigned int) tmp[j] / 255.);
    }

    res.push_back(curr);
  }

  return res;
}
