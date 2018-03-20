#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "../cuda/cuvec.h"

class Perceptron {
private:
	CudaVec weights;
	int nbClass;
	int inputSize;
	float learning_rate;
public:
	Perceptron(int inputSize, int nbClass, float lr);
	int predict();
};

#endif