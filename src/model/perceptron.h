#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include "../utils/data.h"

using namespace std;

class Perceptron {
private:
	float** weights;
	int nbClass;
	int inputSize;
	float learning_rate;
	float* th_vec(float* input);
public:
	Perceptron(int inputSize, int nbClass, float lr);
	int predict(data d);
	float score(vector<data> d);
	void update(vector<data> d);
};

#endif