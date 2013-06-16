#ifndef GRIDINF_INCLUDE_MATRIX_H
#define GRIDINF_INCLUDE_MATRIX_H

#include "utils.h"

#define GINF_MAT_MAX_NUM_DIMS 4

namespace ginf {
	template <typename T>
	class Matrix {
	private:
		// Number of dimensions
		int numDims;

		// The sizes of each dimension
		int dimSize[GINF_MAT_MAX_NUM_DIMS];

		// Memory offsets to correctly convert indices of several dimensions
		// into one single index applied to the flattened array
		int dimIdx[GINF_MAT_MAX_NUM_DIMS];
	public:
		// Expose underlying flattened array for memory copies
		// TODO: use friends
		T *data;

		// Constructor to create a multidimensional matrix with numDim dimensions
		Matrix(int numDims, ...);

		~Matrix();

		// Returns number of dimensions
		int getNumDims();

		// Return size of a specific dimension
		int getSize(int dim);

		// Return total size of the matrix
		int getTotalSize();

		// Return the value at a specific index
		T at(int x, int y, int z = 0, int w = 0);

		// Return a reference to the value at a specific index
		T& get(int x, int y, int z = 0, int w = 0);

		// Copy all the data from a matrix of the same size
		void copyFrom(Matrix<T> *m);

		// Reset all elements to 0
		void clear();
	};
}

#endif
