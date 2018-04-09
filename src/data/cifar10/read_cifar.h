#ifndef READ_CIFAR_H
#define READ_CIFAR_H

#include <vector>
#include <string>

#include "../data.h"

using namespace std;

vector<data> read_batch(string fileName, int nb_img);

#endif
