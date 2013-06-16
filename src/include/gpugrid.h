#ifndef GRIDINF_INCLUDE_GPUGRID_H
#define GRIDINF_INCLUDE_GPUGRID_H

#include "matdim.h"

namespace ginf {
	// Mirrors the Grid class, but uses an implementation that is more suitable for GPUs.
	template <typename T>
	class GpuGrid {
	public:
		int smModel; // The smoothness cost model
		MatDim dimDt, dimSm; // Dimensions of cost matrices
		T *dtCosts; // Data cost matrix
		T *smCosts; // Smoothness cost matrix

// Declare this functions only when compiling with nvcc
#ifdef __CUDACC__
		// Get width/height
		__device__ int getWidth() {
			return dimDt.x;	
		}

		__device__ int getHeight() {
			return dimDt.y;		
		}

		// Get total number of nodes
		__device__ int getNumNodes() {
			return dimDt.x * dimDt.y;		
		}

		// Get number of labels
		__device__ int getNumLabels() {
			return dimDt.z;		
		}

		// Get the cost of labeling (x, y) with label fp
		__device__ T getDataCost(int x, int y, int fp) {
			return dtCosts[dimDt.idx(x, y, fp)];
		}

		// Get the smoothness cost V(fp, fq)
		__device__ T getSmoothnessCost(int fp, int fq) {
			return smCosts[dimSm.idx(fp, fq)];		
		}

		// Returns the cost of a labeling for a particular pixel
		__device__ T getLabelingCost(int *f, int x, int y, int label) {
			T totalCost = getDataCost(x, y, label);
			for (int d = 0; d < GINF_NUM_DIR; d++) {
				int nx = x + dDirX[d], ny = y + dDirY[d];
				if (dimDt.isValid(nx, ny)) {
					totalCost += getSmoothnessCost(label, (int)f[dimDt.idx(nx, ny)]);
				}
			}
			
			return totalCost;
		}
#endif
	};
	
	// Explicit instantiations for template classes
	template class GpuGrid<int>;
	template class GpuGrid<float>;
}

#endif
