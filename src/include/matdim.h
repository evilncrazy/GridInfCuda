#ifndef GRIDINF_INCLUDE_MATDIM_H
#define GRIDINF_INCLUDE_MATDIM_H

namespace ginf {
	// Contains the sizes of each dimension of a matrix 
	struct MatDim {
		int x, y, z;
		MatDim() : x(0), y(0), z(0) { }

// Declare this functions only when compiling with nvcc
#ifdef __CUDACC__
		__device__ MatDim(int xx, int yy, int zz = 0) : x(xx), y(yy), z(zz) { }

		// Convert indices for each dimension into a single index for a flat array
		__device__ int idx(int cx, int cy, int cz = 0, int cw = 0) { return cx + cy * x + cz * x * y + cw * x * y * z; }
		__device__ int operator()(int cx, int cy, int cz = 0, int cw = 0) { return idx(cx, cy, cz, cw); }

		// Returns true if a particular coordinate is between the boundaries of the first two dimensions
		__device__ int isValid(int cx, int cy) { return cx >= 0 && cy >= 0 && cx < x && cy < y; }
		
		// Returns true if a particular coordinate is strictly between the boundaries of the first two dimensions
		__device__ int isValidStrict(int cx, int cy) { return cx > 0 && cy > 0 && cx < x - 1 && cy < y - 1; }
#endif
	};
}

#endif
