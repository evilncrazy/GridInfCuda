#ifndef GRIDINF_INCLUDE_MATDIM_H
#define GRIDINF_INCLUDE_MATDIM_H

namespace ginf {
// Declare this functions only when compiling with nvcc
#ifdef __CUDACC__
	// Contains the sizes of each dimension of a matrix 
	struct MatDim {
		int4 dim; // Dimension sizes
		
		__host__ __device__ MatDim(int x = 0, int y = 0, int z = 0) {
			dim = make_int4(x, y, z, 0);
		}

		// Convert indices for each dimension into a single index for a flat array
		__device__ int idx(int x, int y, int z = 0, int w = 0) {
			return x + y * dim.x + z * dim.x * dim.y + w * dim.x * dim.y * dim.z;
		}
		
		__device__ int operator()(int x, int y, int z = 0, int w = 0) { return idx(x, y, z, w); }

		// Returns true if a particular coordinate is between the boundaries of the first two dimensions
		__device__ int isValid(int x, int y) { return x < dim.x && y < dim.y; }
		
		// Returns true if a particular coordinate is strictly between the boundaries of the first two dimensions
		__device__ int isValidStrict(int x, int y) { return x > 0 && y > 0 && x < dim.x - 1 && y < dim.y - 1; }
	};
#endif
}

#endif
