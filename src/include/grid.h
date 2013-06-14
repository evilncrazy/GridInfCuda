#ifndef GRIDINF_INCLUDE_GRID_H
#define GRIDINF_INCLUDE_GRID_H

#include "utils.h"
#include "matrix.h"

namespace ginf {
	// Represents common smoothness functions
	enum SmoothnessFunctions {
		GINF_SM_EXPLICIT,
		GINF_SM_POTTS,
		GINF_SM_TRUNC_LINEAR,
		GINF_SM_TRUNC_QUAD
	};

	template <typename T>
	class Grid {
	private:
		// dtCosts(x, y, fp) contains the cost of labeling node (x, y) with label fp
		Matrix<T> *dtCosts;

		// smCosts(fp, fq) = smoothness cost V(fp, fq)
		Matrix<T> *smCosts;

		// Keep track of what smoothness cost function we're using
		// This is useful if we want to implement optimizations that only work for
		// particular smoothness functions.
		SmoothnessFunctions smModel;
	public:
		// Constructor to create 2D grid of size (w, h), with n labels
		Grid(int w, int h, int n);

		~Grid();

		// Get width/height
		int getWidth();
		int getHeight();

		// Get number of labels
		int getNumLabels();

		// Get total number of nodes
		int getNumNodes();
		
		// Get the smoothness function model
		SmoothnessFunctions getSmModel();

		// Get/set the cost of labeling (x, y) with label fp
		T getDataCost(int x, int y, int fp);
		void setDataCost(int x, int y, int fp, T cost);

		// Get the smoothness cost V(fp, fq)
		T getSmoothnessCost(int fp, int fq);

		// Explicitly set the smoothness cost V(fp, fq)
		void setSmoothnessCost(int fp, int fq, T cost);

		// Use the Potts model as the smoothness cost function
		// with V(fp, fq) = lambda if fp != fq, 0 otherwise
		void useSmoothnessPotts(T lambda);

		// Use truncated linear model as the smoothness cost function
		// with V(fp, fq) = min(scale * |fp - fq|, d)
		void useSmoothnessTruncLinear(T scale, T d);

		// Use truncated quadratic model as smoothness cost function
		// with V(fp, fq) = min(scale * (fp - fq)^2, d)
		void useSmoothnessTruncQuad(T scale, T d);

		// Returns the total energy cost of a labeling
		T getLabelingCost(Matrix<int> *f);

		// Returns the cost of a labeling for a particular pixel
		T getLabelingCost(Matrix<int> *f, int x, int y, int label);

		// Returns the data and smoothness cost matrices
		Matrix<T> *getDataCosts();
		Matrix<T> *getSmoothnessCosts();
	};
}

#endif
