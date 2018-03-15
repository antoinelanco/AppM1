#include <iostream>

#include "utils/res.h"
#include "read_cifar.h"
#include "cuda/test.h"

int main(int argc, char** argv) {
	vector<img> data = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 10000);

	int toprint = data.size() - 1;
	cout << data.size() << endl;
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 32; j++) {
			cout << (((unsigned int)data[toprint].red[i * 32 + j]) < 122u ? ". " : "# ");
		}
		cout << endl;
	}
	cout << "label : " << (unsigned int) data[toprint].label << endl;
	cout << endl;

	int n = 11;
	float* deviceData = mallocCuda(n);
	initVec(deviceData, n);
	float* hostData = copyToHost(deviceData, n);
	for (int i = 0; i < n; i++) {
		cout << hostData[i] << endl;
	}
	return 0;
} 