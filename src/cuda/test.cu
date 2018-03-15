#include "test.h"


__global__ void init_vec(float* vec) {
	int tid = threadIdx.x;
	vec[tid] = 15.;
}

int getNbThreadPerBlock(int device) {
	int value;
	cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, device);
	return value;
}

float* mallocCuda(int nbElt) {
	float* res;
	cudaMalloc(&res, sizeof(float) * nbElt);
	return res;
}

float* copyToHost(float* deviceData, int nbElt) {
	float* res = (float*) malloc(sizeof(float) * nbElt);
	cudaMemcpy(res, deviceData, sizeof(float) * nbElt, cudaMemcpyDeviceToHost);
	return res;
}

void initVec(float* deviceData, int nbElt) {
	int nbThread = getNbThreadPerBlock(0);
	init_vec<<<1, nbElt>>>(deviceData);
}