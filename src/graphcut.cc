#include "include/graphcut.h"

#include <cstdio>

#include <algorithm>
#include <queue>

namespace ginf {
	template <typename T>
	void graphCutConstructGraph(Grid<T> *grid, Matrix<int> *f, Matrix<int> *height, Matrix<T> *excess, Matrix<T> *capacity, int alpha, int beta) {
		for (int y = 0; y < grid->getHeight(); y++) {
			for (int x = 0; x < grid->getWidth(); x++) {
				// Check if (x, y) is in the alpha-beta partition
				if (f->at(x, y) == alpha || f->at(x, y) == beta) {
					T alphaCost = 0, betaCost = 0;
					for (int d = 0; d < GINF_NUM_DIR; d++) {
						int nx = x + dirX[d], ny = y + dirY[d];
						if (GINF_IS_VALID_NODE(nx, ny, grid->getWidth(), grid->getHeight())) {
							// Check if this neighbour (nx, ny) is in the alpha-beta partition
							if (f->at(nx, ny) == alpha || f->at(nx, ny) == beta) {
								// Capacity of this edge is V(alpha, beta)
								capacity->get(x, y, d) = grid->getSmoothnessCost(alpha, beta);
							} else {
								// There is no edge between these two nodes
								capacity->get(x, y, d) = 0;

								alphaCost += grid->getSmoothnessCost(alpha, f->at(nx, ny));
								betaCost += grid->getSmoothnessCost(beta, f->at(nx, ny));
							}
						} else {
							// No edge here
							capacity->get(x, y, d) = 0;
						}
					}

					// Calculate the capacity of t-links
					excess->get(x, y) = grid->getDataCost(x, y, alpha) + alphaCost;
					capacity->get(x, y, GINF_DIR_SINK) = grid->getDataCost(x, y, beta) + betaCost;
					
					// All nodes start with height 0
					height->get(x, y) = 0;
				}
			}
		}
	}
	
	template <typename T>
	void graphCutFindReachable(Grid<T> *grid, Matrix<T> *residue, Matrix<int> *reachable, int x, int y) {
		reachable->get(x, y) = true;

		for (int d = 0; d < 4; d++) {
			int nx = x + dirX[d], ny = y + dirY[d];
			if (GINF_IS_VALID_NODE(nx, ny, grid->getWidth(), grid->getHeight()) &&
				residue->at(nx, ny, GINF_OPP_DIR(d)) && !reachable->at(nx, ny)) {
				graphCutFindReachable(grid, residue, reachable, nx, ny);
			}
		}
	}

	void graphCutLabelFromFlow(Matrix<int> *f, Matrix<int> *reachable, int alpha, int beta) {
		for (int y = 0; y < f->getSize(1); y++) {
			for (int x = 0; x < f->getSize(0); x++) {
				if (f->at(x, y) == alpha || f->at(x, y) == beta) {
					if (reachable->at(x, y)) {
						f->get(x, y) = alpha;
					} else {
						f->get(x, y) = beta;
					}
				}
			}
		}
	}
	
	template <typename T>
	int graphCutCountActive(Grid<T> *grid, Matrix<int> *height, Matrix<T> *excess) {
		int active = 0;
		
		for (int y = 0; y < grid->getHeight(); y++) {
			for (int x = 0; x < grid->getWidth(); x++) {
				// Check there's excess
				if (excess->at(x, y) && height->at(x, y) < grid->getNumNodes() + 2) {
					active++;
				}
			}
		}

		return active;
	}

	template <typename T>
	void graphCutGlobalRelabel(Grid<T> *grid, Matrix<int> *height, Matrix<T> *residue) {
		// Create a queue to perform Breadth First Search
		std::queue< std::pair<int, int> > q;

		// Matrix to store the new relabeled heights
		Matrix<int> h(2, grid->getWidth(), grid->getHeight());

		for (int y = 0; y < grid->getHeight(); y++) {
			for (int x = 0; x < grid->getWidth(); x++) {
				// Find nodes that have residue edges to the sink
				if (residue->at(x, y, GINF_DIR_SINK)) {
					h.get(x, y) = 1;

					// Push it on the queue
					q.push(std::make_pair(x, y));
				}
			}
		}

		while (!q.empty()) {
			int x = q.front().first, y = q.front().second;
			q.pop();

			// Find residual edges in each direction
			for (int d = 0; d < GINF_NUM_DIR; d++) {
				int nx = x + dirX[d], ny = y + dirY[d];
				if (GINF_IS_VALID_NODE(nx, ny, grid->getWidth(), grid->getHeight())) {
					// Take reversed edged directions and see if there's a residual edge
					if (residue->at(nx, ny, GINF_OPP_DIR(d)) && h.at(nx, ny) == 0) {
						h.get(nx, ny) = h.at(x, y) + 1;

						// Push this to the queue
						q.push(std::make_pair(nx, ny));
					}
				}
			}
		}

		// For nodes not reacheable from the sink, we'll just give them the maximum height
		for (int y = 0; y < grid->getHeight(); y++) {
			for (int x = 0; x < grid->getWidth(); x++) {
				if (h.get(x, y) == 0) {
					h.get(x, y) = GINF_MAX(height->at(x, y), grid->getNumNodes() + 2);
				}
			}
		}

		// Copy the new heights over
		height->copyFrom(&h);
	}
	
	template <typename T>
	void graphCutPushToPixel(Grid<T> *grid, Matrix<T> *excess, Matrix<T> *residue, int x, int y, int d) {		
		int nx = x + dirX[d], ny = y + dirY[d];

		// We can push either all of the residue flow or just the excess,
		// depending on which one is smaller
		T flow = GINF_MIN(residue->at(x, y, d), excess->at(x, y));

		// Adjust the excess and residue values for this node and the
		// neighbour, as we push 'flow' from this node to the neighbour
		excess->get(x, y) -= flow;
		excess->get(nx, ny) += flow;
		
		residue->get(x, y, d) -= flow;
		residue->get(nx, ny, GINF_OPP_DIR(d)) += flow;
	}
	
	template <typename T>
	void graphCutPush(Grid<T> *grid, Matrix<int> *height, Matrix<T> *excess, Matrix<T> *residue, int x, int y) {
		if (excess->at(x, y) && height->at(x, y) < grid->getNumNodes() + 2) {
			// Try to push to the sink
			if (residue->at(x, y, GINF_DIR_SINK) && height->at(x, y) == 1) {
				T flow = GINF_MIN(residue->at(x, y, GINF_DIR_SINK), excess->at(x, y));

				excess->get(x, y) -= flow;
				residue->get(x, y, GINF_DIR_SINK) -= flow;
			}
			
			// Push to neighbours
			for (int d = 0; d < GINF_NUM_DIR; d++) {
				if (residue->at(x, y, d) && height->at(x, y) == height->at(x + dirX[d], y + dirY[d]) + 1) {
					graphCutPushToPixel(grid, excess, residue, x, y, d);
				}
			}
		}
	}
	
	template <typename T>
	void graphCutRelabel(Grid<T> *grid, Matrix<int> *height, Matrix<T> *excess, Matrix<T> *residue, int x, int y) {
		if (excess->at(x, y) && height->at(x, y) < grid->getNumNodes() + 2) {
			if (residue->at(x, y, GINF_DIR_SINK)) {
				height->get(x, y) = 1;
			} else {
				int minHeight = grid->getNumNodes() + 2;
				for (int d = 0; d < GINF_NUM_DIR; d++) {
					if (residue->at(x, y, d)) {
						minHeight = GINF_MIN(minHeight, height->at(x + dirX[d], y + dirY[d]));
					}
				}
				
				height->get(x, y) = minHeight + 1;
			}
		}
	}

	template <typename T>
	void decodeAlphaBeta(Grid<T> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result) {
		// We'll need some matrices to store excess, residues and height
		Matrix<T> excess(2, grid->getWidth(), grid->getHeight());
		Matrix<int> height(2, grid->getWidth(), grid->getHeight());
		Matrix<T> residue(3, grid->getWidth(), grid->getHeight(), GINF_NUM_DIR + 2);

		// Start with initial configuration
		result->copyFrom(initial);
		
		// Boolean matrix indicating whether a node can reach the sink in the residue graph
		Matrix<int> reachable(2, grid->getWidth(), grid->getHeight());
		
		// Loop until there are no more label changes	
		T minCost = grid->getLabelingCost(result);
		
		// Create a temp matrix to store new labelings
		Matrix<int> newResult(2, grid->getWidth(), grid->getHeight());
		newResult.copyFrom(initial);
		
		bool success = true;
		for (int t = 0; t < numIters && success; t++) {
			success = false;

			// Loop through each pair of labels
			for (int alpha = 0; alpha < grid->getNumLabels(); alpha++) {
				for (int beta = alpha + 1; beta < grid->getNumLabels(); beta++) {
					// Now, construct the graph with alpha and beta as terminal nodes
					graphCutConstructGraph(grid, result, &height, &excess, &residue, alpha, beta);

					// Run push-relabel until convergence
					while (true) {
						// Check if we're done
						if (graphCutCountActive(grid, &height, &excess) == 0)
							break;
						
						// Global relabeling optimization
						graphCutGlobalRelabel(grid, &height, &residue);

						// Run normal push and relabel
						for (int i = 0; i < GINF_GRAPHCUT_NUM_ITERS_PER_GLOBAL_RELABEL; i++) {
							for (int y = 0; y < grid->getHeight(); y++) {
								for (int x = 0; x < grid->getWidth(); x++) {
									graphCutPush(grid, &height, &excess, &residue, x, y);
								}
							}
							
							for (int y = 0; y < grid->getHeight(); y++) {
								for (int x = 0; x < grid->getWidth(); x++) {
									graphCutRelabel(grid, &height, &excess, &residue, x, y);
								}
							}
						}
					}

					// Find the labeling induced by the residue graph
					reachable.clear();
					for (int y = 0; y < grid->getHeight(); y++) {
						for (int x = 0; x < grid->getWidth(); x++) {	
							if (residue.at(x, y, GINF_DIR_SINK)) {
								graphCutFindReachable(grid, &residue, &reachable, x, y);
							}	
						}
					}
					
					// Give each of the nodes their new labels after alpha-beta swap
					graphCutLabelFromFlow(&newResult, &reachable, alpha, beta);
					
					// Calculate the cost of this labeling
					T cost = grid->getLabelingCost(&newResult);
					
					if (cost < minCost) {
						minCost = cost;
						result->copyFrom(&newResult);
						
						success = true;
					} else {
						newResult.copyFrom(result);
					}
				}
			}
		}
	}
	
	template void decodeAlphaBeta<int>(Grid<int>*, Matrix<int>*, int, Matrix<int>*);
}
