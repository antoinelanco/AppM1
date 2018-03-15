#include "cuvec.h"
#include "cudautils.h"
#include <stdlib.h>
#include <iostream>

using namespace std;

__global__ void dotCuda(float* tmp, float* t1, float* t2, int size) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	tmp[tid] = t1[tid] * t2[tid];

	__syncthreads();

	int mididx = size / 2;

	while (tid <= mididx && mididx != 0) {
		tmp[tid] += tmp[tid * 2];
		mididx /= 2;
		__syncthreads();
	}
}

__global__ void init_vec(float* vec, float value) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
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
	exit(0);
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

float CudaVec::dot(CudaVec other) {
	if (this->size != other.size) {
		cout << "Uncompatible size !" << endl;
		exit(0);
	}

	float* tmp;
	cudaMalloc(&tmp, sizeof(float) * this->size);

	

	int nbBlX = getNbBlockDimX(0);
	int thrPBl = getNbThreadPerBlock(0);

	if (this->size <= thrPBl) {
		dotCuda<<<1, this->size>>>(tmp, this->cudaptr, other.cudaptr, this->size);
	} else {
		int neededBl = this->size / nbBlX;
		if (neededBl <= nbBlX) {
			dotCuda<<<neededBl, thrPBl>>>(tmp, this->cudaptr, other.cudaptr, this->size);
		} else {
			cout << "Unimplemented nbBlNeeded : " << neededBl << " (" << nbBlX << " available)" << endl;
			exit(0);
		}
	}
	float result;
	cudaMemcpy(&result, tmp, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	cudaFree(tmp);
	return result;
}