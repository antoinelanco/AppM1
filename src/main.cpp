#include <iostream>

#include "utils/res.h"
#include "read_cifar.h"
#include "utils/data.h"
#include "model/perceptron.h"

int main(int argc, char** argv) {
	vector<img_brute> d = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 10000);

	int toprint = d.size() - 1;
	cout << d.size() << endl;
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 32; j++) {
			cout << (((unsigned int)d[toprint].red[i * 32 + j]) < 122u ? ". " : "# ");
		}
		cout << endl;
	}
	cout << "label : " << (unsigned int) d[toprint].label << endl;

	vector<data> batch_data = transform_to_data(d);
	toprint = batch_data.size() - 2;
	cout << batch_data.size() << endl;
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 32; j++) {
			float moy = 0.f;
			moy = max(moy, batch_data[toprint].features[i * 32 + j]);
			moy = max(moy, batch_data[toprint].features[i * 32 + j + 1024]);
			moy = max(moy, batch_data[toprint].features[i * 32 + j + 2048]);
			cout << ( moy < 0.5f ? ". " : "# ");
		}
		cout << endl;
	}
	cout << "label : " << batch_data[toprint].label << endl;

	vector<img_brute> img_test = read_batch(getResFolder() + "/cifar-10-batches-bin/data_batch_1.bin", 1000);
	vector<data> data_test = transform_to_data(img_test);

	Perceptron p(32*32*3, 10, 0.1);
	for (int i = 0; i < 30; i++) {
		cout << "Epoch " << i << ", score : " << p.score(batch_data);
		p.update(batch_data);
		int nbErr = 0;
		for (int i = 0; i < 1000; i++) {
			int predict = p.predict(data_test[i]);
			if (predict != data_test[i].label)
				nbErr++;
		}
		cout << ", nbErr : " << nbErr << " / " << 1000 << endl;
	}

	return 0;
}
