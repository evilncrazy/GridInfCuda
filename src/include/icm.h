#ifndef GRIDINF_INCLUDE_ICM_H
#define GRIDINF_INCLUDE_ICM_H

#include "core.h"
#include "gpu.h"

// CUDA block size for ICM
#define GINF_ICM_BLOCK_SIZE 16

namespace ginf {
	// Decode using synchronous ICM (each node changes labels at the same time)
	template <typename T>
	void decodeIcmSync(Grid<T> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result);

	// Decode using synchronous ICM on CUDA
	template <typename T>
	void gpuDecodeIcm(Grid<T> *grid, Matrix<int> *initial, int numIters, Matrix<int> *results);
	
	// Decode using asynchronous ICM (nodes change labels sequentially)
	template <typename T>
	void decodeIcmAsync(Grid<T> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result);
}

#endif
