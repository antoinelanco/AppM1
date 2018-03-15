int getNbThreadPerBlock(int device) {
	int value;
	cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, device);
	return value;
}

int getNbBlockDimX(int device) {
	int value;
	cudaDeviceGetAttribute(&value, cudaDevAttrMaxBlockDimX, device);
	return value;
}