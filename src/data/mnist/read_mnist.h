#ifndef READ_MNIST_H
#define READ_MNIST_H

#include <vector>
#include <string>

#include "../data.h"

using namespace std;

vector<data> read_mnist(string dataFileName, string labelFileName, int nbToLoad);

#endif
