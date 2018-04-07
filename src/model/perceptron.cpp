#include "perceptron.h"
#include <limits>
#include <math.h>
#include <iostream>
#include <tuple>
#include <algorithm>

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
	float max = -numeric_limits<float>::max();
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

bool pairCompare(pair<int, int>& firstElem, pair<int, int>& secondElem) {
  return firstElem.first < secondElem.first;
}

float Perceptron::score(vector<data> d) {
	float err = 0.f;
	float total = 0.f;
	// vector<pair<int, int>> res;
	vector<int> count(this->weights.size(), 0);
	for (data curr_d : d) {
		// pair<int, int> a(this->predict(curr_d), curr_d.label);
		int pred = this->predict(curr_d);
		if (pred != curr_d.label)
			err++;
		count[pred]++;
		total++;
		// res.push_back(a);
	}
	for (int i = 0; i < this->weights.size(); i++) {
		cout << i << " : " << count[i] << endl;
	}
	// sort(res.begin(), res.end(), pairCompare);
	// for (int i = 0; i < res.size(); i++) {
	// 	cout << res[i].first << " " << res[i].second << endl;
	// }
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
		// if (this->predict(curr_d) == curr_d.label)
		//  	continue;
		float* g = this->th_vec(curr_d.features);
		for (int k = 0; k < this->weights.size(); k++) {
			float etiquette = curr_d.label == k ? 1.f : -1.f;
			for (int j = 0; j < this->weights[k].size(); j++) {
				this->weights[k][j] += -this->learning_rate * (g[k] - etiquette) * curr_d.features[j];
			}
		}
	}
	this->learning_rate *= 0.9;
}

// ------------------------------------- 2 ----------------------------------------------------
Perceptron2::Perceptron2(int inputSize, int nbClass, float lr) {
	this->inputSize = inputSize;
	this->nbClass = nbClass;
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

int Perceptron2::predict(data d) {

	int argmax = 0;
	float max = 0.;

	for (int i = 0; i < this->nbClass; i++) {
		float sum = 0.f;
		std::vector<float> w = this->weights[i];
		for (int j = 0; j < this->inputSize; j++) {
			sum += d.features[j] * w[j];
		}
		if (sum > max) {
			// std::cout << "ici" << '\n';
			max = sum;
			argmax = i;
		}
	}
	//std::cout << "ici" << max << '\n';
	return argmax;
}

float Perceptron2::score(vector<data> d) {
	float err = 0.f;
	float total = 0.f;
	for (data curr_d : d) {
	//std::cout << this->predict(curr_d) << " " << curr_d.label << '\n';

		if (this->predict(curr_d) != curr_d.label){
			err++;
		}
		total++;
	}
	return err / total;
}


void Perceptron2::update(vector<data> d) {
	for (data curr_d : d) {
		int pred = this->predict(curr_d);
		if ( pred != curr_d.label){
			for (int k = 0; k < this->nbClass; k++) {
				for (int j = 0; j < this->inputSize; j++) {
					this->weights[k][j] += this->learning_rate * (curr_d.label - pred) * curr_d.features[j];
				}
			}
		}
	}
}
