#ifndef GRIDINF_INCLUDE_GPUGRID_H
#define GRIDINF_INCLUDE_GPUGRID_H

#include "matdim.h"

namespace ginf {
	// Mirrors the Grid class, but uses an implementation that is more suitable for GPUs.
	template <typename T>
	class GpuGrid {
	public:
// Declare this functions only when compiling with nvcc
#ifdef __CUDACC__
		int smModel; // The smoothness cost model
		MatDim dimDt, dimSm; // Dimensions of cost matrices
		T *dtCosts; // Data cost matrix
		T *smCosts; // Smoothness cost matrix

		// Get width/height
		__device__ int getWidth() {
			return dimDt.dim.x;	
		}

		__device__ int getHeight() {
			return dimDt.dim.y;		
		}

		// Get total number of nodes
		__device__ int getNumNodes() {
			return getWidth() * getHeight();		
		}

		// Get number of labels
		__device__ int getNumLabels() {
			return dimDt.dim.z;		
		}

		// Get the cost of labeling (x, y) with label fp
		__device__ T getDataCost(int x, int y, int fp) {
			return dtCosts[dimDt(x, y, fp)];
		}

		// Get the smoothness cost V(fp, fq)
		__device__ T getSmoothnessCost(int fp, int fq) {
			return smCosts[dimSm(fp, fq)];
		}
#endif
	};
	
	// Explicit instantiations for template classes
	template class GpuGrid<int>;
	template class GpuGrid<float>;
}

#endif
