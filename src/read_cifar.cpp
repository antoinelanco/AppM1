#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "read_cifar.h"
#include "cuda/test.h"

vector<img> read_batch(string fileName, int nb_img) {
	vector<img> res;
	ifstream in(fileName.c_str());

    if (!in) {
        cout << "Error during opening models file" << endl;
        exit(0);
    }

    for (int i = 0; i < nb_img; ++i) {
    	img curr;
    	in.read(&curr.label, 1);
    	in.read(curr.red, 1024);
    	in.read(curr.green, 1024);
    	in.read(curr.blue, 1024);

    	res.push_back(curr);
    }

	return res;
}