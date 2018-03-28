#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "read_cifar.h"

vector<img_brute> read_batch(string fileName, int nb_img) {
	vector<img_brute> res;
	ifstream in(fileName.c_str());

    if (!in) {
        cout << "Error during opening models file" << endl;
        exit(0);
    }

    for (int i = 0; i < nb_img; ++i) {
    	img_brute curr;
    	in.read(&curr.label, 1);
    	in.read(curr.red, 1024);
    	in.read(curr.green, 1024);
    	in.read(curr.blue, 1024);

    	res.push_back(curr);
    }

	return res;
}

vector<data> transform_to_data(vector<img_brute> v) {
    vector<data> res;

    for (img_brute img : v) {
        data curr;
        curr.features = new float[1024 * 3];
        for (int i = 0; i < 1024; i++) {
            curr.features[i * 3 + 0] = ((float)(unsigned int) img.red[i]) / 255.f;
						curr.features[i * 3 + 1] = ((float)(unsigned int) img.green[i]) / 255.f;
						curr.features[i * 3 + 2] = ((float)(unsigned int) img.blue[i]) / 255.f;
        }
        curr.label = img.label;
        res.push_back(curr);
    }
    return res;
}
