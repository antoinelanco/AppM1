#include <iostream>

#include "utils/res.h"
#include "data/read_cifar.h"
#include "data/read_mnist.h"
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
	int nbPatch = 4;

	cout << "Loading Data..." <<endl;
	vector<img_brute> d = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 1000);
	vector<data> batch_data = transform_to_data(d);

	cout << "Split data..." << endl;
	vector<data> splittedData = split(batch_data, nbPatch);
	int N = 2048;
	cout << "Learn K-Means..." << endl;
	pair<vector<data>, K_means > resGather = trainKMeansDataFeatures(splittedData, N, 100, nbPatch);

	cout << "Learn Perceptron..." << endl;
	Perceptron p(N * nbPatch, 10, 0.1);
	int nbEpoch = 100;
	for (int i = 0; i < nbEpoch; i++) {
		cout << '\r' << ((i+1.)/nbEpoch)*100 << "%" << std::flush;
		p.update(resGather.first);
	}

	cout << "\nTest..." << endl;

	vector<img_brute> img_test = read_batch(getResFolder() + "/cifar-10-batches-bin/test_batch.bin", 1000);
	vector<data> data_test = transform_to_data(img_test);

	vector<data> splittedTestImg = split(data_test, nbPatch);
	vector<data> featuresData = gatherDataFeatures(resGather.second, splittedTestImg, N, nbPatch);

	cout << "Score : " << p.score(featuresData) << endl;
}

void test() {
	cout << "Loading Data..." <<endl;
	vector<img_brute> d = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 1);
	vector<data> batch_data = transform_to_data(d);

	int nbPatch = 16; // Must be power of 2;
	vector<data> splittedData = split(batch_data, nbPatch);
	cout << splittedData.size() << endl;
	for (int i = 0; i < splittedData.size(); i++) {
		cout << splittedData[i].features.size() << endl;
	}
}

void mnist() {
	cout << "Loading data..." << endl;
	vector<data> trainData = read_mnist(getResFolder() + "/mnist/train-images-idx3-ubyte",
	 getResFolder() + "/mnist/train-labels-idx1-ubyte",
	 1000);

	vector<data> testData = read_mnist(getResFolder() + "/mnist/t10k-images-idx3-ubyte",
	 getResFolder() + "/mnist/t10k-labels-idx1-ubyte",
	 1000);
	cout << "Perceptron training..." << endl;
	Perceptron p(28 * 28, 10, 0.1);
 	int nbEpoch = 100;
 	for (int i = 0; i < nbEpoch; i++) {
 		cout << '\r' << ((i+1.)/nbEpoch)*100 << "%" << flush;
 		p.update(trainData);
 	}
	cout << endl;

	cout << "Score : " << p.score(testData) << endl << endl;

	cout << "K-Means training..." << endl;
	K_means k(10, trainData);
	k.proc(10);
	cout << "Score : " << k.loss(testData) << endl;
}

int main(int argc, char** argv) {
	//approcheDesBoss();
	//test();
	mnist();
	return 0;
}
