#include "include/matrix.h"

#include <cstdarg>
#include <cstring>
#include <cstdlib>

namespace ginf {
	template <typename T>
	Matrix<T>::Matrix(int numDims, ...) {
		this->numDims = numDims;

		va_list args;
		va_start(args, numDims);

		// Set each of the dimension sizes and calculate the dimension offset indices
		for (int i = 0; i < numDims; i++) {
			dimSize[i] = va_arg(args, int);
			dimIdx[i] = dimSize[i] * (i ? dimIdx[i - 1] : 1);
		}

		// Allocate enough memory
		data = (T *)malloc(sizeof(T) * getTotalSize());

		va_end(args);
	
		// Set all elements to 0 by default
		clear();
	}

	template <typename T>
	Matrix<T>::~Matrix() {
		free(data); data = NULL;
	}

	template <typename T>
	int Matrix<T>::getNumDims() {
		return numDims;
	}

	template <typename T>
	int Matrix<T>::getSize(int dim) {
		return dimSize[dim];
	}

	template <typename T>
	int Matrix<T>::getTotalSize() {
		// Total size is the product of the sizes of all dimensions
		// which is the last entry in dimIdx
		return dimIdx[numDims - 1];
	}
	
	template <typename T>
	T Matrix<T>::at(int x, int y, int z, int w) { return data[x + dimIdx[0] * y + dimIdx[1] * z + dimIdx[2] * w]; }
	
	template <typename T>
	T& Matrix<T>::get(int x, int y, int z, int w) { return data[x + dimIdx[0] * y + dimIdx[1] * z + dimIdx[2] * w]; }

	template <typename T>
	void Matrix<T>::copyFrom(Matrix *m) {
		memcpy(data, m->data, sizeof(T) * getTotalSize());
	}

	template <typename T>
	void Matrix<T>::clear() {
		memset(data, 0, sizeof(T) * getTotalSize());
	}

	// Explicit instantiations for template classes
	template class Matrix<int>;
	template class Matrix<float>;
}
