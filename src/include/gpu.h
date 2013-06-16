#ifndef GRIDINF_INCLUDE_GPU_GPU_H
#define GRIDINF_INCLUDE_GPU_GPU_H

#ifdef __CUDACC__
namespace ginf {
	// Direction offsets that can be used on the device
	__constant__ int dDirX[] = {0, 1, 0, -1};
	__constant__ int dDirY[] = {1, 0, -1, 0};
}
#endif

#include "core.h"
#include "matdim.h"

#include "gpugrid.h"

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

// Macro to calculate the number of thread blocks needed in a
// single dimension, given the number of threads in that dimension (x)
// and the size of each block in that dimension (s)
#define GINF_DIM(x, s) ((x - 1) / s + 1)

// Macro that declares (x, y) variables for a specific thread
#define GINF_DECL_X int x = blockIdx.x * blockDim.x + threadIdx.x
#define GINF_DECL_Y int y = blockIdx.y * blockDim.y + threadIdx.y

// Macro that declares alternating x coordinates for a specific thread, so that
// no two threads will be working on adjacent pixels
#define GINF_DECL_ALT_X(iter) int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + ((blockIdx.y * blockDim.y + threadIdx.y + iter) & 1)

// Attempt to call a CUDA API function. If it fails, then the program exits with an error message.
#define GINF_SAFE_CALL(call) { cudaAssert(call, __FILE__, __LINE__); } 

namespace ginf {
	class Gpu {
	public:
		// Stops execution of program if there was a CUDA error
		void cudaAssert(const cudaError err, const char *file, const int line) {
			if (cudaSuccess != err) {
				fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
					file, line, cudaGetErrorString(err) );
				exit(1);
			}
		}
	
		// Create a copy of a matrix on the device
		template <typename T>
		T *createRawMat(Matrix<T> *hostM) {
			T *dMat = NULL;

			// Allocate memory on device
			GINF_SAFE_CALL(cudaMalloc((void **)&dMat, sizeof(T) * hostM->getTotalSize()));

			// Copy matrix to device
			GINF_SAFE_CALL(cudaMemcpy(dMat, hostM->data, sizeof(T) * hostM->getTotalSize(), cudaMemcpyHostToDevice));
		
			return dMat;
		}
		
		// Create an empty matrix of a particular length on the device
		template <typename T>
		T *createRawMat(int length) {
			T *dMat = NULL;

			// Allocate memory on device
			GINF_SAFE_CALL(cudaMalloc((void **)&dMat, sizeof(T) * length));

			// Set all the elements to 0
			GINF_SAFE_CALL(cudaMemset(dMat, 0, sizeof(T) * length));
		
			return dMat;
		}
		
		// Create a copy of a grid on the device
		template <typename T>
		GpuGrid<T> *createGrid(Grid<T> *hostG) {
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

		// Copy a host matrix to the device
		template <typename T>
		void copyRawMat(T *deviceM, Matrix<T> *hostM) {
			GINF_SAFE_CALL(cudaMemcpy(deviceM, hostM->data, sizeof(T) * hostM->getTotalSize(), cudaMemcpyHostToDevice));
		}
		
		// Copy a device matrix to the host
		template <typename T>
		void copyRawMat(Matrix<T> *hostM, T *deviceM) {
			GINF_SAFE_CALL(cudaMemcpy(hostM->data, deviceM, sizeof(T) * hostM->getTotalSize(), cudaMemcpyDeviceToHost));
		}
		
		// Copy a device matrix to another device matrix
		template <typename T>
		void copyRawMat(T *deviceDst, T *deviceSrc, int len) {
			GINF_SAFE_CALL(cudaMemcpy(deviceDst, deviceSrc, sizeof(T) * len, cudaMemcpyDeviceToDevice));
		}
		
		// Free device memory
		template <typename T>
		void free(T *dPtr) {
			GINF_SAFE_CALL(cudaFree(dPtr));
		}
		
		template <typename T>
		void free(GpuGrid<T> *dGrid) {
			GINF_SAFE_CALL(cudaFree(dGrid));
		}
		
	};
}

#endif
