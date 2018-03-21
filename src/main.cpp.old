#include <iostream>

#include "utils/res.h"
#include "read_cifar.h"
#include "cuda/test.h"
#include "cuda/cuvec.h"

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
	free(hostData);

	n = 32 * 32 * 3 * 10;
	CudaVec cuvec1(n);
	cuvec1.fill(1.f);
	hostData = cuvec1.toHost();
	float acc = 0.f;
	for (int i = 0; i < n; i++) {
		acc += hostData[i];
	}
	cout << n << " = " << acc << endl;

	float* tmp = new float[n];
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 32 * 32 * 3; j++) {
			tmp[i * 3072 + j] = 0.5f + i;
		}
	}
	CudaVec cuvec2(tmp, n);
	//cuvec2.fill(0.f);

	clock_t begin = clock();
	float* res = cuvec1.dot(cuvec2, 32 * 32 * 3);
  	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  	cout << "CUDA dot for " << n << " elements : " << elapsed_secs << "s" << endl;
  	for (int i = 0; i < 10; i++) {
		cout << (0.5f + i) * n / 10 << " = " << res[i] << endl;
	}

	float* a = (float*) malloc(sizeof(float) * n);
	float* b = (float*) malloc(sizeof(float) * n);
	for (int i = 0; i < n; i++) {
		a[i] = b[i] = 1.f;
	}

	float sum = 0.f;
	begin = clock();
	for (int i = 0; i < n; i++) {
		sum += a[i] * b[i];
	}
	end = clock();
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  	cout << "CPU dot for " << n << " elements : " << elapsed_secs << "s" << endl;
  	cout << sum << endl;

	return 0;
} 