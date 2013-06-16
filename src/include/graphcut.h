#ifndef GRIDINF_INCLUDE_GRAPHCUT_H
#define GRIDINF_INCLUDE_GRAPHCUT_H

#include "core.h"
#include "gpu.h"

// Number of iterations of push relabel before a global relabeling operation
#define GINF_GRAPHCUT_NUM_ITERS_PER_GLOBAL_RELABEL 10
#define GINF_GRAPHCUT_BLOCK_SIZE 16

namespace ginf {
	enum TerminalDirections {
		GINF_DIR_SOURCE = GINF_NUM_DIR,
		GINF_DIR_SINK
	};

	// Decode using alpha beta swap
	template <typename T>
	void decodeAlphaBeta(Grid<T> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result);
	
	// Decode using alpha beta swap on the GPU
	template <typename T>
	void gpuDecodeAlphaBeta(Grid<T> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result);
	
	// Find nodes reachable from the source in the residue graph
	template <typename T>
	void graphCutFindReachable(Grid<T> *grid, Matrix<T> *residue, Matrix<int> *reachable, int x, int y);
	
	// Given a max flow residue graph, find labeling induced by the minimum cut
	void graphCutLabelFromFlow(Matrix<int> *f, Matrix<int> *reachable, int alpha, int beta);

	// Relabel the heights of each node with its shortest distance to the sink
	// Returns true if no more nodes are active
	template <typename T>
	bool graphCutGlobalRelabel(Grid<T> *grid, Matrix<int> *height, Matrix<T> *excess, Matrix<T> *residue);
}

#endif
