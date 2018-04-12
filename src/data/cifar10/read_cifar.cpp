#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "read_cifar.h"

vector<data> read_batch(string fileName, int nb_img) {
		vector<data> res;
		ifstream in(fileName.c_str());

    if (!in) {
        cout << "Error during opening models file" << endl;
        exit(0);
    }

    for (int i = 0; i < nb_img; ++i) {
			char* red = new char[1024];
			char* green = new char[1024];
			char* blue = new char[1024];
			char label;
    	in.read(&label, 1);
    	in.read(red, 1024);
    	in.read(green, 1024);
    	in.read(blue, 1024);

			data curr;
			curr.features = vector<float>();
			for (int j = 0; j < 1024; j++) {
					float r = (unsigned char) red[j] / 255.f;
					float g = (unsigned char) green[j] / 255.f;
					float b = (unsigned char) blue[j] / 255.f;
					curr.features.push_back(r);
					curr.features.push_back(g);
					curr.features.push_back(b);
			}
			delete red;
			delete green;
			delete blue;
			curr.label = label;
			res.push_back(curr);
    }
	return res;
}
