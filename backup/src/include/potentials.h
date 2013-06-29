#ifndef GRIDINF_INCLUDE_POTENTIALS_H
#define GRIDINF_INCLUDE_POTENTIALS_H

#include "gpu.h"

namespace ginf {
#ifdef __CUDACC__
	template <typename T>
	class PairwisePot {
	private:
		GpuGrid<T> *g;
		int m;
		int2 params;
	
	public:
		__device__ PairwisePot(GpuGrid<T> *grid) {
			m = grid->smModel;
			
			switch(m) {
				case GINF_SM_POTTS:
				case GINF_SM_TRUNC_LINEAR:
					params.x = grid->getSmoothnessCost(0, 1); // linear scaling constant
					params.y = grid->getSmoothnessCost(0, grid->getNumLabels() - 1); // truncation constant
			}
			
			g = grid;
		}
		
		__device__ int operator()(int fp, int fq) {
			switch(m) {
				case GINF_SM_POTTS:
				case GINF_SM_TRUNC_LINEAR:
					return GINF_MIN(params.x * GINF_ABS(fp - fq), params.y);
				default:
					return g->getSmoothnessCost(fp, fq);
			}
		}
	};
#endif
}

#endif
