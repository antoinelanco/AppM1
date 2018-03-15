#include "cuvec.h"
#include "cudautils.h"
#include <stdlib.h>
#include <iostream>

using namespace std;

__global__ void init_vec(float* vec, float value) {
	int tid = blockIdx.x *blockDim.x + threadIdx.x;
	vec[tid] = value;
}

CudaVec::CudaVec(int size) {
	this->size = size;
	if (this->size % 32 != 0)
		exit(0);
	cudaMalloc(&this->cudaptr, sizeof(float) * this->size);
	cudaMemset(this->cudaptr, 0, sizeof(float) * this->size);
}

void CudaVec::fill(float value) {
	int nbBlX = getNbBlockDimX(0);
	int thrPBl = getNbThreadPerBlock(0);

	if (this->size <= thrPBl) {
		init_vec<<<1, this->size>>>(this->cudaptr, value);
		return;
	}

	int neededBl = this->size / nbBlX;
	if (neededBl <= nbBlX) {
		init_vec<<<neededBl, thrPBl>>>(this->cudaptr, value);
		return;
	}
	cout << "Unimplemented nbBlNeeded : " << neededBl << " (" << nbBlX << " available)" << endl;
}

void CudaVec::free() {
	cudaFree(this->cudaptr);
}

float* CudaVec::toHost() {
	float* hostVec = (float*) malloc(sizeof(float) * this->size);
	cudaMemcpy(hostVec, this->cudaptr, sizeof(float) * this->size, cudaMemcpyDeviceToHost);
	return hostVec;
}

int CudaVec::getSize() {
	return this->size;
}