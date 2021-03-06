#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <string>
#include "../data/data.h"

using namespace std;

class Perceptron {
private:
	vector<vector<float>> weights;
	int nbClass;
	int inputSize;
	float learning_rate;
	float* th_vec(vector<float> input);
public:
	Perceptron(string fileName);
	Perceptron(int inputSize, int nbClass, float lr);
	int predict(data d);
	float score(vector<data> d);
	void scoreFile(vector<data> d);
	void update(vector<data> d);
	void toFile();
};

class Perceptron2{
private:
	vector<vector<float>> weights;
	int nbClass;
	int inputSize;
	float learning_rate;
public:
	Perceptron2(int inputSize, int nbClass, float lr);
	int predict(data d);
	float score(vector<data> d);
	void update(vector<data> d);
};

#endif
