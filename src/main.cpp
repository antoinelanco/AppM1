#include <iostream>

#include "utils/res.h"
#include "data/read_cifar.h"
#include "data/data.h"
#include "model/perceptron.h"
#include "model/k_means.h"

int main(int argc, char** argv) {
	vector<img_brute> d = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 10000);

	vector<data> batch_data = transform_to_data(d);
	// int toprint = batch_data.size() - 1;
	// cout << batch_data.size() << endl;
	// for (int i = 0; i < 32; i++) {
	// 	for (int j = 0; j < 32; j++) {
	// 		float moy = 0.f;
	// 		moy = max(moy, batch_data[toprint].features[(i * 32 + j) * 3]);
	// 		moy = max(moy, batch_data[toprint].features[(i * 32 + j) * 3 + 1]);
	// 		moy = max(moy, batch_data[toprint].features[(i * 32 + j) * 3 + 2]);
	// 		cout << ( moy < 0.5f ? ". " : "# ");
	// 	}
	// 	cout << endl;
	// }
	// cout << "label : " << batch_data[toprint].label << endl;

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
	return 0;
}
