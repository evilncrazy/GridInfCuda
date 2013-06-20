#include "../include/icm.h"
#include "../include/potentials.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace ginf {
	template <typename T>
	__device__ T gpuIcmGetLabelingCost(GpuGrid<T> *grid, PairwisePot<T> pot, int4 labeling, int label) {
		GINF_DECL_X; GINF_DECL_Y;
		
		return grid->getDataCost(x, y, label) +
			pot(label, labeling.x) +
			pot(label, labeling.y) +
			pot(label, labeling.z) +
			pot(label, labeling.w);
	}
	
	template <typename T>
	__global__ void gpuIcmDecodeKernel(GpuGrid<T> *grid, int *result, int *old) {
		GINF_DECL_X; GINF_DECL_Y;
		MatDim dim(grid->getWidth(), grid->getHeight());
		PairwisePot<T> pot(grid);
		
		if (dim.isValidStrict(x, y)) {
			// Store the labels of neighbours in register memory
			int4 labels = make_int4(
				old[dim(x, y + 1)],
				old[dim(x + 1, y)],
				old[dim(x, y - 1)],
				old[dim(x - 1, y)]
			);
			
			// Find the label that gives the minimum energy
			T minCost = gpuIcmGetLabelingCost(grid, pot, labels, 0);
			int minLabel = 0;
			
			for (int i = 1; i < grid->getNumLabels(); i++) {		
				T cost = gpuIcmGetLabelingCost(grid, pot, labels, i);
				if (cost < minCost) {
					minLabel = i;
					minCost = cost;
				}
			}
			
			// Update this node's state
			result[dim(x, y)] = minLabel;
		}
	}
	
	template <typename T>
	void gpuDecodeIcm(Grid<T> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result) {
		Gpu gpu;
	
		// Start with initial labeling
		result->copyFrom(initial);

		// Allocate GPU memory and transfer data
		int *dResult = gpu.createRawMat(result),
		    *dOld = gpu.createRawMat(result); // double buffer the labeling matrix
		GpuGrid<T> *dGrid = gpu.createGrid(grid);

		// Prepare block sizes
		dim3 blockSize(GINF_ICM_BLOCK_SIZE, GINF_ICM_BLOCK_SIZE);
		dim3 gridSize(GINF_DIM(grid->getWidth(), GINF_ICM_BLOCK_SIZE), 
					  GINF_DIM(grid->getHeight(), GINF_ICM_BLOCK_SIZE));

		// For each iteration, launch the ICM kernel
		for (int i = 0; i < numIters; i++) {
			gpuIcmDecodeKernel<<<gridSize, blockSize>>>(dGrid, dResult, dOld);
			gpu.copyRawMat(dOld, dResult, result->getTotalSize());
		}
		
		// Transfer the results back to CPU memory
		gpu.copyRawMat(result, dResult);
		
		// Free GPU memory
		gpu.free(dResult);
		gpu.free(dOld);
		gpu.free(dGrid);
	}
	
	template void gpuDecodeIcm<float>(Grid<float> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result);
	template void gpuDecodeIcm<int>(Grid<int> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result);
}
