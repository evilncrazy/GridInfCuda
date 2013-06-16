#ifndef GRIDINF_INCLUDE_BP_H
#define GRIDINF_INCLUDE_BP_H

#include "core.h"
#include "gpu.h"

// CUDA block size for belief propagation
#define GINF_BP_BLOCK_SIZE 16

namespace ginf {
	// Decode using synchronous loopy belief propagation
	template <typename T>
	void decodeBpSync(Grid<T> *grid, int numIters, Matrix<int> *result);

	// Decode using synchronous loopy belief propagation on GPU
	template <typename T>
	void gpuDecodeBp(Grid<T> *grid, int numIters, Matrix<int> *result);

	// Decode using hierarchical belief propagation
	template <typename T>
	void decodeHbp(Grid<T> *grid, int numLevels, int numItersPerLevel, Matrix<int> *result);
	
	// Decode using hierarchical belief propagation on GPU
	template <typename T>
	void gpuDecodeHbp(Grid<T> *grid, int numLevels, int numItersPerLevel, Matrix<int> *result);
}

#endif
