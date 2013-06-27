#include "include/grid.h"

namespace ginf {
	template <typename T>
	Grid<T>::Grid(int w, int h, int n) {
		// Create the cost matrices
		dtCosts = new Matrix<T>(3, w, h, n);
		smCosts = new Matrix<T>(2, n, n);

		// Initially, the smoothness costs are all 0, therefore it can be
		// considered as a Potts model with zero lambda
		smModel = GINF_SM_POTTS;
		
		width = w; height = h; numLabels = n;
	}

	template <typename T>
	Grid<T>::~Grid() {
		delete dtCosts; dtCosts = 0;
		delete smCosts; smCosts = 0;
	}

	template <typename T>
	int Grid<T>::getWidth() {
		return width;
	}

	template <typename T>
	int Grid<T>::getHeight() {
		return height;
	}

	template <typename T>
	int Grid<T>::getNumLabels() {
		return numLabels;
	}

	template <typename T>
	int Grid<T>::getNumNodes() {
		return getWidth() * getHeight();
	}

	template <typename T>
	SmoothnessFunctions Grid<T>::getSmModel() {
		return smModel;
	}

	template <typename T>
	T Grid<T>::getDataCost(int x, int y, int f) {
		return dtCosts->at(x, y, f);
	}

	template <typename T>
	void Grid<T>::setDataCost(int x, int y, int f, T cost) {
		dtCosts->get(x, y, f) = cost;
	}

	template <typename T>
	T Grid<T>::getSmoothnessCost(int fp, int fq) {
		return smCosts->at(fp, fq);
	}
	
	template <typename T>
	void Grid<T>::setSmoothnessCost(int fp, int fq, T cost) {
		smModel = GINF_SM_EXPLICIT;
		smCosts->get(fp, fq) = cost;
	}

	template <typename T>
	void Grid<T>::useSmoothnessPotts(T lambda) {
		smModel = GINF_SM_POTTS;

		for (int fp = 0; fp < getNumLabels(); fp++) {
			for (int fq = 0; fq <= fp; fq++) {
				smCosts->get(fp, fq) = smCosts->get(fq, fp) = lambda * (fp != fq);
			}
		}
	}

	template <typename T>
	void Grid<T>::useSmoothnessTruncLinear(T scale, T d) {
		smModel = GINF_SM_TRUNC_LINEAR;

		for (int fp = 0; fp < getNumLabels(); fp++) {
			for (int fq = 0; fq <= fp; fq++) {
				smCosts->get(fp, fq) = smCosts->get(fq, fp) = GINF_MIN(scale * GINF_ABS(fp - fq), d);
			}
		}
	}

	template <typename T>
	void Grid<T>::useSmoothnessTruncQuad(T scale, T d) {
		smModel = GINF_SM_TRUNC_QUAD;

		for (int fp = 0; fp < getNumLabels(); fp++) {
			for (int fq = 0; fq <= fp; fq++) {
				smCosts->get(fp, fq) = smCosts->get(fq, fp) = GINF_MIN(scale * GINF_SQ(fp - fq), d);
			}
		}
	}

	template <typename T>
	T Grid<T>::getLabelingCost(Matrix<int> *f) {
		// For each node (x, y), we'll calculate its data cost and its smoothness cost
		// with two of its neighbours (horizontal and vertical, to avoid double counting)
		T totalCost = 0;

		int w = getWidth(), h = getHeight();
		for (int y = 1; y < h - 1; y++) {
			for (int x = 1; x < w - 1; x++) {
				// Count the data cost D
				totalCost += getDataCost(x, y, f->at(x, y));
				
				// For its neighbours in the north and east direction, count the
				// smoothness costs
				int nx = x + dirX[GINF_DIR_N], ny = y + dirY[GINF_DIR_N];
				totalCost += getSmoothnessCost(f->at(x, y), f->at(nx, ny));

				int ex = x + dirX[GINF_DIR_E], ey = y + dirY[GINF_DIR_E];
				totalCost += getSmoothnessCost(f->at(x, y), f->at(ex, ey));
			}
		}

		return totalCost;
	}

	template <typename T>
	T Grid<T>::getLabelingCost(Matrix<int> *f, int x, int y, int label) {
		// For each direction, we calculate the smoothness cost with that neighbour
		T totalCost = getDataCost(x, y, label);
		for (int d = 0; d < GINF_NUM_DIR; d++) {
			totalCost += getSmoothnessCost(label, f->at(x + dirX[d], y + dirY[d]));
		}

		return totalCost;
	}

	template <typename T>
	Matrix<T> *Grid<T>::getDataCosts() {
		return dtCosts;
	}

	template <typename T>
	Matrix<T> *Grid<T>::getSmCosts() {
		return smCosts;
	}
	
	// Explicit instantiations for template classes
	template class Grid<int>;
	template class Grid<float>;
}
