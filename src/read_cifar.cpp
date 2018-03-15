#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>

using namespace std;

struct img {
	char red[1024];
	char green[1024];
	char blue[1024];
	char label;
};

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

int main(int argc, char** argv) {
	vector<img> data = read_batch("./cifar-10-batches-bin/data_batch_1.bin", 10000);
	cout << data.size() << endl;
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 32; j++) {
			cout << (((unsigned int)data[data.size() - 1].red[i * 32 + j]) < 122u ? ". " : "# ");
		}
		cout << endl;
	}
	cout << (unsigned int) data[data.size() - 1].label << endl;
	return 0;
} 