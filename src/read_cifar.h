#ifndef READ_CIFAR_H
#define READ_CIFAR_H

#include <vector>
#include <string>

using namespace std;

struct img {
	char red[1024];
	char green[1024];
	char blue[1024];
	char label;
};

vector<img> read_batch(string fileName, int nb_img);

#endif