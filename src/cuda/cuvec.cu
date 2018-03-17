#include "cuvec.h"
#include "cudautils.h"
#include <stdlib.h>
#include <iostream>

using namespace std;

__global__ void reduce0(float* g_odata, float* g_idata1, float* g_idata2) {
	extern __shared__ float sdata[];
	// each thread loads one element from global to shared mem

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata1[i] * g_idata2[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) { 
		g_odata[blockIdx.x] = sdata[0];
		//atomicAdd(g_odata, sdata[0]);
	}
}

__global__ void dotCuda(float* tmp, float* t1, float* t2, int size, int fake_size) {
	//unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	tmp[i] = t1[i] * t2[i];
	__syncthreads();

	int mididx = size / 2;

	//bool needCorrect = mididx % 2 == 1;

	while (i < mididx) {
		tmp[i] += tmp[i + mididx];
		mididx /= 2;
		__syncthreads();
	}
	//atomicAdd(tmp, p);
}

__global__ void init_vec(float* vec, float value) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	vec[tid] = value;
}

CudaVec::CudaVec(int size) {
	this->size = size;
	if (this->size % 2 != 0)
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
	//tmp[0] = 0;

	int fake_size = (int) pow(2, ceil(log(this->size)/log(2)));

	int nbBlX = getNbBlockDimX(0);
	int thrPBl = getNbThreadPerBlock(0);

	int neededBl = fake_size / nbBlX;

	float* tmp;
	cudaMalloc(&tmp, sizeof(float) * neededBl);

	if (fake_size <= thrPBl) {
		dotCuda<<<1, fake_size>>>(tmp, this->cudaptr, other.cudaptr, this->size);
		//reduce0<<<1, this->size>>>(tmp, this->cudaptr, other.cudaptr);
	} else {
		
		//cout << "pb appel, started : " << (neededBl * thrPBl) << ", curr : " << this->size << endl;

		if (neededBl <= nbBlX) {
			dotCuda<<<neededBl, thrPBl>>>(tmp, this->cudaptr, other.cudaptr, this->size);
			//reduce0<<<neededBl, thrPBl>>>(tmp, this->cudaptr, other.cudaptr);
		} else {
			cout << "Unimplemented nbBlNeeded : " << neededBl << " (" << nbBlX << " available)" << endl;
			exit(0);
		}
	}

	/*float* result = (float*) malloc(sizeof(float) * neededBl);
	// Le resultat est dans tmp[0]
	cudaMemcpy(result, tmp, sizeof(float) * neededBl, cudaMemcpyDeviceToHost);
	cudaFree(tmp);

	float sum = 0.f;
	for (int i = 0; i < neededBl; i++) {
		sum += result[i];
	}

	return sum;*/
	float result = 0.f;
	cudaMemcpy(&result, tmp, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	cudaFree(tmp);
	return result;
}