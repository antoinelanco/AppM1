#ifndef CUVEC_H
#define CUVEC_H

class CudaVec {
private:
	float* cudaptr;
	int size;
public:
	CudaVec(int size);
	void fill(float value);
	void free();
	float* toHost();
	int getSize();
	float dot(CudaVec other);
};

#endif