#include <iostream>

#include "utils/res.h"
#include "read_cifar.h"

int main(int argc, char** argv) {
	vector<img> data = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 10000);
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