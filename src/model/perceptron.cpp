#include "perceptron.h"
#include <limits>
#include <math.h>
#include <iostream>

#include <vector>

Perceptron::Perceptron(int inputSize, int nbClass, float lr) {
	this->learning_rate = lr;

	this->weights = vector<vector<float>>();

	for (int i = 0; i < nbClass; i++) {
		vector<float> tmp;
		for (int j = 0; j < inputSize; j++) {
			tmp.push_back(0.f);
		}
		this->weights.push_back(tmp);
	}
}

int Perceptron::predict(data d) {
	int argmax = -1;
	float max = numeric_limits<float>::min();
	for (int i = 0; i < this->weights.size(); i++) {
		float sum = 0.f;
		for (int j = 0; j < this->weights[i].size(); j++) {
			sum += d.features[j] * this->weights[i][j];
		}
		if (max < sum) {
			max = sum;
			argmax = i;
		}
	}
	return argmax;
}

float Perceptron::score(vector<data> d) {
	float err = 0.f;
	float total = 0.f;
	for (data curr_d : d) {
		if (this->predict(curr_d) != curr_d.label)
			err++;
		total++;
	}
	return err / total;
}

float* Perceptron::th_vec(vector<float> input) {
	float* res = new float[input.size()];
	for (int k = 0; k < this->weights.size(); k++) {
		res[k] = 0.f;
		for (int j = 0; j < this->weights[k].size(); j++) {
			res[k] += this->weights[k][j] * input[j];
		}
		res[k] = tanh(res[k]);
	}
	return res;
}

void Perceptron::update(vector<data> d) {
	for (data curr_d : d) {
		if (this->predict(curr_d) == curr_d.label)
			continue;
		float* g = this->th_vec(curr_d.features);
		for (int k = 0; k < this->weights.size(); k++) {
			float etiquette = curr_d.label == k ? 1.f : -1.f;
			for (int j = 0; j < this->weights[k].size(); j++) {
				this->weights[k][j] += -this->learning_rate * (g[k] - etiquette) * curr_d.features[j];
			}
		}
	}
}
