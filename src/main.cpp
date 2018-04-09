#include <iostream>

#include "utils/res.h"
#include "data/cifar10/read_cifar.h"
#include "data/mnist/read_mnist.h"
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
	vector<data> batch_data = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 10000);

	vector<data> data_test = read_batch(getResFolder() + "/cifar-10-batches-bin/test_batch.bin", 1000);

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
	int nbPatch = 16;

	cout << "Loading Data..." <<endl;
	vector<data> batch_data = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 1000);

	cout << "Split data..." << endl;
	vector<data> splittedData = split(batch_data, nbPatch);
	int N = 128;
	cout << "Learn K-Means..." << endl;

	//pair<vector<data>, K_means> resGather = trainKMeansDataFeatures(splittedData, N, 30, nbPatch);
	K_Means_2 k(N, 32 * 32 * 3 / nbPatch, splittedData);
	int nbIter = 20;
	for (int i = 0; i < nbIter; i++) {
		k.update(splittedData);
		cout << '\r' << 100 * (int) (i + 1.) / nbIter << "%" << flush;
	}
	cout << endl;
	vector<data> newData = gatherDataFeatures(k, splittedData, N, nbPatch);

	// float min = 0.;
	// float max = 0.;
	// for (size_t i = 0; i < 1000; i++) {
	// 	for (size_t j = 0; j < N * nbPatch; j++) {
	// 		int tmp = newData[i].features[j];
	// 		if (tmp > max) max = tmp;
	// 		if (tmp < min) min = tmp;
	// 	}
	// }
	// cout << "max : " << max << ", min : " << min << endl;

	cout << "Learn Perceptron..." << endl;
	Perceptron p(N * nbPatch, 10, 0.01);
	int nbEpoch = 100;
	for (int i = 0; i < nbEpoch; i++) {
		cout << '\r' << 100 * (int) (i + 1.) / nbEpoch << "%" << std::flush;
		p.update(newData);
	}

	cout << "\nTest..." << endl;

	vector<data> data_test = read_batch(getResFolder() + "/cifar-10-batches-bin/test_batch.bin", 1000);

	vector<data> splittedTestImg = split(data_test, nbPatch);
	vector<data> featuresData = gatherDataFeatures(k, splittedTestImg, N, nbPatch);
	cout << "Score : " << p.score(featuresData) << endl;
}

void test() {
	cout << "Loading Data..." <<endl;
	vector<data> trainData = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 1000);

	vector<data> data_test = read_batch(getResFolder() + "/cifar-10-batches-bin/test_batch.bin", 1000);

	float min = 0.;
	float max = 0.;
	for (size_t i = 0; i < 1000; i++) {
		for (size_t j = 0; j < 32 * 32 * 3; j++) {
			int tmp = trainData[i].features[j];
			if (tmp > max) max = tmp;
			if (tmp < min) min = tmp;
		}
	}
	cout << "max : " << max << ", min : " << min << endl;

	cout << "Perceptron training cifar..." << endl;
	Perceptron p(28 * 28, 10, 0.1);
 	int nbEpoch = 100;
 	for (int i = 0; i < nbEpoch; i++) {
 		p.update(trainData);
		cout << '\r' << ((i+1.)/nbEpoch)*100 << "%" << flush;
 	}
	cout << endl << "Score (trainData) : " << p.score(trainData) << endl;
	cout << "Score (testData) : " << p.score(data_test) << endl << endl;
}

void mnist() {
	cout << "Loading data..." << endl;
	vector<data> trainData = read_mnist(
		getResFolder() + "/mnist/train-images-idx3-ubyte",
	 	getResFolder() + "/mnist/train-labels-idx1-ubyte",
	 	1000);

	vector<data> testData = read_mnist(
		getResFolder() + "/mnist/t10k-images-idx3-ubyte",
	 	getResFolder() + "/mnist/t10k-labels-idx1-ubyte",
	 	1000);

	cout << "Perceptron training mnist..." << endl;
	Perceptron p(28 * 28, 10, 0.1);
 	int nbEpoch = 100;
 	for (int i = 0; i < nbEpoch; i++) {
 		cout << '\r' << ((i+1.)/nbEpoch)*100 << "%" << flush;
 		p.update(trainData);
 	}
	cout << endl << "Score : " << p.score(testData) << endl << endl;

	cout << "K-Means training mnist..." << endl;
	/*K_means k(10, trainData);
	k.proc(10);*/
	K_Means_2 k(10, 28 * 28, trainData);
	int nbIter = 30;
	for (int i = 0; i < nbIter; i++) {
		k.update(trainData);
		cout << '\r' << 100 * (int) (i + 1.) / nbIter << "%" << flush;
	}
	cout << endl << "Score : " << k.score(testData) << endl;
}

int main(int argc, char** argv) {
	approcheDesBoss();
	//test();
	//mnist();
	return 0;
}
