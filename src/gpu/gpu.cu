#include "../include/gpu/gpu.h"

#include <cstdio>

namespace ginf {
	template <typename T>
	T *Gpu::createRawMat(Matrix<T> *hostM) {
		T *dMat = NULL;

		// Allocate memory on device
		GINF_SAFE_CALL(cudaMalloc((void **)&dMat, sizeof(T) * hostM->getTotalSize()));

		// Copy matrix to device
		GINF_SAFE_CALL(cudaMemcpy(dMat, hostM->data, sizeof(T) * hostM->getTotalSize(), cudaMemcpyHostToDevice));
		
		return dMat;
	}
	
	template <typename T>
	T *Gpu::createRawMat(int length) {
		T *dMat = NULL;

		// Allocate memory on device
		GINF_SAFE_CALL(cudaMalloc((void **)&dMat, sizeof(T) * length));

		// Set all the elements to 0
		GINF_SAFE_CALL(cudaMemset(dMat, 0, sizeof(T) * length));
		
		return dMat;
	}

	template <typename T>
	GpuGrid<T> *Gpu::createGrid(Grid<T> *hostG) {
		GpuGrid<T> *dGrid = NULL;

		// We'll need to use a temporary GpuGrid to hold device addresses of
		// the cost matrices
		GpuGrid<T> tempGrid;
		tempGrid.smModel = hostG->getSmModel();
		tempGrid.dimDt = MatDim(hostG->getWidth(), hostG->getHeight(), hostG->getNumLabels());
		tempGrid.dimSm = MatDim(hostG->getNumLabels(), hostG->getNumLabels());

		// Allocate grid on device
		// Use the tempGrid to hold device addresses for data and smoothness cost
		// matrices, and then copy tempGrid to our actual dGrid
		GINF_SAFE_CALL(cudaMalloc((void **)&(tempGrid.dtCosts),
			sizeof(T) * hostG->getDataCosts()->getTotalSize()));
		GINF_SAFE_CALL(cudaMalloc((void **)&(tempGrid.smCosts),
			sizeof(T) * hostG->getSmCosts()->getTotalSize()));
		GINF_SAFE_CALL(cudaMalloc((void **)&dGrid, sizeof(GpuGrid<T>)));

		// Copy grid to device
		GINF_SAFE_CALL(cudaMemcpy(tempGrid.dtCosts, hostG->getDataCosts()->data,
			sizeof(T) * hostG->getDataCosts()->getTotalSize(), cudaMemcpyHostToDevice));
		GINF_SAFE_CALL(cudaMemcpy(tempGrid.smCosts, hostG->getSmCosts()->data,
			sizeof(T) * hostG->getSmCosts()->getTotalSize(), cudaMemcpyHostToDevice));
		GINF_SAFE_CALL(cudaMemcpy(dGrid, &tempGrid, sizeof(GpuGrid<T>), cudaMemcpyHostToDevice));

		return dGrid;
	}
	
	template <typename T>
	void Gpu::copyRawMat(T *deviceM, Matrix<T> *hostM) {
		GINF_SAFE_CALL(cudaMemcpy(deviceM, hostM->data, sizeof(T) * hostM->getTotalSize(), cudaMemcpyHostToDevice));
	}
	
	template <typename T>
	void Gpu::copyRawMat(Matrix<T> *hostM, T *deviceM) {
		GINF_SAFE_CALL(cudaMemcpy(hostM->data, deviceM, sizeof(T) * hostM->getTotalSize(), cudaMemcpyDeviceToHost));
	}
	
	template <typename T>
	void Gpu::copyRawMat(T *deviceDst, T *deviceSrc, int len) {
		GINF_SAFE_CALL(cudaMemcpy(deviceDst, deviceSrc, sizeof(T) * len, cudaMemcpyDeviceToDevice));
	}
	
	template <typename T>
	void Gpu::free(T *dPtr) {
		GINF_SAFE_CALL(cudaFree(dPtr));
	}
	
	template <typename T>
	void Gpu::free(GpuGrid<T> *dGrid) {
		GINF_SAFE_CALL(cudaFree(dGrid));
	}

	void cudaAssert(const cudaError err, const char *file, const int line) {
	    if (cudaSuccess != err) {
			fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
			        file, line, cudaGetErrorString(err) );
			exit(1);
	    }
	}
}
