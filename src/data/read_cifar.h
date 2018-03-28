#ifndef READ_CIFAR_H
#define READ_CIFAR_H

#include <vector>
#include <string>

#include "data.h"

using namespace std;

struct img_brute {
	char red[1024];
	char green[1024];
	char blue[1024];
	char label;
};

vector<img_brute> read_batch(string fileName, int nb_img);

vector<data> transform_to_data(vector<img_brute> v);

#endif
