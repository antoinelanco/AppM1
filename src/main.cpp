#include <iostream>

#include "utils/res.h"
#include "data/read_cifar.h"
#include "data/data.h"
#include "model/perceptron.h"
#include "model/k_means.h"
#include "data/gather_data.h"
#include "data/splitimage.h"

void printImg(data toprint) {
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 32; j++) {
			float moy = 0.f;
			moy = max(moy, toprint.features[(i * 32 + j) * 3]);
			moy = max(moy, toprint.features[(i * 32 + j) * 3 + 1]);
			moy = max(moy, toprint.features[(i * 32 + j) * 3 + 2]);
			cout << ( moy < 0.5f ? ". " : "# ");
		}
		cout << endl;
	}
	cout << "label : " << toprint.label << endl;
}

void approcheNaive() {
	vector<img_brute> d = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 10000);

	vector<data> batch_data = transform_to_data(d);

	vector<img_brute> img_test = read_batch(getResFolder() + "/cifar-10-batches-bin/test_batch.bin", 1000);
	vector<data> data_test = transform_to_data(img_test);
	//
	Perceptron p(32*32*3, 10, 0.1);
	for (int i = 0; i < 30; i++) {
		cout << "Epoch " << i << ", ";
		p.update(batch_data);
		cout << "taux d'erreur : " << p.score(data_test) << endl;
	}

	cout << "K-Means :" << endl;

	K_means k(10,batch_data);
	k.proc(20);
	std::cout << "Error rate on test set : " << k.loss(data_test)*100 << "%" << '\n';
}

void approcheDesBoss() {
	cout << "Loading Data..." <<endl;
	vector<img_brute> d = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 1000);
	vector<data> batch_data = transform_to_data(d);

	cout << "Split data..." << endl;
	vector<data> splittedData = split(batch_data);
	int N = 30;
	cout << "Learn K-Means..." << endl;
	pair<vector<data>, K_means > resGather = trainKMeansDataFeatures(splittedData, N, 10);

	cout << "Learn Perceptron..." << endl;
	Perceptron p(N * 4, 10, 0.1);
	int nbEpoch = 10;
	for (int i = 0; i < nbEpoch; i++) {
		cout << "Epoch " << i << endl;
		p.update(resGather.first);
	}

	cout << "Test" << endl;

	vector<img_brute> img_test = read_batch(getResFolder() + "/cifar-10-batches-bin/test_batch.bin", 1000);
	vector<data> data_test = transform_to_data(img_test);

	vector<data> splittedTestImg = split(data_test);
	vector<data> featuresData = gatherDataFeatures(resGather.second, splittedTestImg, N);

	cout << "Score : " << p.score(featuresData) << endl;
}

int main(int argc, char** argv) {
	approcheDesBoss();
	return 0;
}
