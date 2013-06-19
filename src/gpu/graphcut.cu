#include "../include/graphcut.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace ginf {
	template <typename T>
	__device__ void gpuGraphCutPushToPixel(bool iter, GpuGrid<T> *grid, T *excess, T *residue, int d) {
		GINF_DECL_ALT_X(iter); GINF_DECL_Y;
		MatDim dim(grid->getWidth(), grid->getHeight());
		
		int nx = x + dDirX[d], ny = y + dDirY[d];

		// We can push either all of the residue flow or just the excess,
		// depending on which one is smaller
		T flow = GINF_MIN(residue[dim.idx(x, y, d)], excess[dim.idx(x, y)]);
	
		// Adjust the excess and residue using atomics to prevent RAW hazards
		atomicSub(&excess[dim.idx(x, y)], flow);
		atomicAdd(&excess[dim.idx(nx, ny)], flow);

		atomicSub(&residue[dim.idx(x, y, d)], flow);
		atomicAdd(&residue[dim.idx(nx, ny, GINF_OPP_DIR(d))], flow);
	}
	
	template <typename T>
	__global__ void gpuGraphCutPushKernel(bool iter, GpuGrid<T> *grid, int *height, T *excess, T *residue) {
		GINF_DECL_ALT_X(iter); GINF_DECL_Y;
		MatDim dim(grid->getWidth(), grid->getHeight());

		if (dim.isValid(x, y) && excess[dim.idx(x, y)] && height[dim.idx(x, y)] < grid->getNumNodes() + 2) {
			// Try to push to the sink
			if (residue[dim.idx(x, y, GINF_DIR_SINK)] && height[dim.idx(x, y)] == 1) {
				T flow = GINF_MIN(residue[dim.idx(x, y, GINF_DIR_SINK)], excess[dim.idx(x, y)]);

				excess[dim.idx(x, y)] -= flow;
				residue[dim.idx(x, y, GINF_DIR_SINK)] -= flow;
			}
		
			// Push to neighbours
			for (int d = 0; d < 4; d++) {
				if (residue[dim.idx(x, y, d)] && height[dim.idx(x, y)] == height[dim.idx(x + dDirX[d], y + dDirY[d])] + 1) {
					gpuGraphCutPushToPixel(iter, grid, excess, residue, d);
				}
			}
		}
	}
	
	template <typename T>
	__global__ void gpuGraphCutRelabelKernel(bool iter, GpuGrid<T> *grid, int *height, T *excess, T *residue) {
		GINF_DECL_ALT_X(iter); GINF_DECL_Y;
		MatDim dim(grid->getWidth(), grid->getHeight());

		if (dim.isValid(x, y) && excess[dim.idx(x, y)] && height[dim.idx(x, y)] < grid->getNumNodes() + 2) {
			if (residue[dim.idx(x, y, GINF_DIR_SINK)]) {
				height[dim.idx(x, y)] = 1;
			} else {
				int minHeight = grid->getNumNodes() + 2;
				for (int d = 0; d < 4; d++) {
					if (residue[dim.idx(x, y, d)]) {
						minHeight = GINF_MIN(minHeight, height[dim.idx(x + dDirX[d], y + dDirY[d])]);
					}
				}
				height[dim.idx(x, y)] = minHeight + 1;
			}
		}
	}

	template <typename T>
	__global__ void gpuGraphCutConstructGraphKernel(GpuGrid<T> *grid, int *f, int *height, T *excess, T *capacity, int alpha, int beta) {
		GINF_DECL_X; GINF_DECL_Y;
		MatDim dim(grid->getWidth(), grid->getHeight());

		// Check if (x, y) is in the alpha-beta partition
		if (dim.isValid(x, y) && f[dim.idx(x, y)] == alpha || f[dim.idx(x, y)] == beta) {
			int alphaCost = 0, betaCost = 0;
			for (int d = 0; d < 4; d++) {
				int nx = x + dDirX[d], ny = y + dDirY[d];
				if (dim.isValid(nx, ny)) {
					// Check if this neighbour (nx, ny) is in the alpha-beta partition
					if (f[dim.idx(nx, ny)] == alpha || f[dim.idx(nx, ny)] == beta) {
						// Capacity of this edge is V(alpha, beta)
						capacity[dim.idx(x, y, d)] = grid->getSmoothnessCost(alpha, beta);
					} else {
						// There is no edge between these two nodes
						capacity[dim.idx(x, y, d)] = 0;

						alphaCost += grid->getSmoothnessCost(alpha, f[dim.idx(nx, ny)]);
						betaCost += grid->getSmoothnessCost(beta, f[dim.idx(nx, ny)]);
					}
				} else {
					// No edge here
					capacity[dim.idx(x, y, d)] = 0;
				}
			}

			// Calculate the capacity of t-links
			capacity[dim.idx(x, y, GINF_DIR_SOURCE)] = grid->getDataCost(x, y, alpha) + alphaCost;
			capacity[dim.idx(x, y, GINF_DIR_SINK)] = grid->getDataCost(x, y, beta) + betaCost;
			
			// Saturate t-link from source
			excess[dim.idx(x, y)] = capacity[dim.idx(x, y, GINF_DIR_SOURCE)];
		
			// Reset height to 0
			height[dim.idx(x, y)] = 0;
		}
	}

	template <typename T>
	void gpuDecodeAlphaBeta(Grid<T> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result) {
		Gpu gpu;
	
		// We'll need some matrices to store excess, residues and height
		Matrix<T> excess(2, grid->getWidth(), grid->getHeight());
		Matrix<int> height(2, grid->getWidth(), grid->getHeight());
		Matrix<T> residue(3, grid->getWidth(), grid->getHeight(), GINF_NUM_DIR + 2);

		// Start with the intial labeling
		result->copyFrom(initial);
		
		// Boolean matrix indicating whether a node is reachable from the source in the residue graph
		Matrix<int> reachable(2, grid->getWidth(), grid->getHeight());

		// Create a temp matrix to store new labelings
		Matrix<int> newResult(2, grid->getWidth(), grid->getHeight());
		newResult.copyFrom(initial);

		// Allocate GPU memory and transfer data
		T *dExcess = gpu.createRawMat(&excess),
		  *dResidue = gpu.createRawMat(&residue);
		int *dResult = gpu.createRawMat(result),
		  *dHeight = gpu.createRawMat(&height);
		
		GpuGrid<T> *dGrid = gpu.createGrid(grid);

		// Prepare block sizes
		dim3 blockSize(GINF_GRAPHCUT_BLOCK_SIZE, GINF_GRAPHCUT_BLOCK_SIZE);
		dim3 gridSize(GINF_DIM(grid->getWidth(), GINF_GRAPHCUT_BLOCK_SIZE), 
					  GINF_DIM(grid->getHeight(), GINF_GRAPHCUT_BLOCK_SIZE));

		// Loop until there are no more label changes
		bool success = true;
		for (int t = 0; t < numIters && success; t++) {
			success = false;

			// Loop through each pair of labels
			T minCost = grid->getLabelingCost(result);
			for (int alpha = 1; alpha < grid->getNumLabels(); alpha++) {
				for (int beta = 0; beta < alpha; beta++) {
					// Now, construct the graph with alpha and beta as terminal nodes
					gpuGraphCutConstructGraphKernel<<<gridSize, blockSize>>>(dGrid, dResult, dHeight, dExcess, dResidue, alpha, beta);

					// We need to keep this data synced between CPU and GPU
					// every time we perform global relabeling
					gpu.copyRawMat(&height, dHeight);
					gpu.copyRawMat(&excess, dExcess);
					gpu.copyRawMat(&residue, dResidue);

					// Run push-relabel until convergence
					int iters = 0;
					while (true) {
						// Global relabeling optimization on the CPU
						if (graphCutGlobalRelabel(grid, &height, &excess, &residue)) {
							// No more active nodes, we're done
							break;
						}
						
						// Copy relabeled height data back to the GPU
						gpu.copyRawMat(dHeight, &height);

						// Run normal push and relabel on the GPU
						for (int i = 0; i < GINF_GRAPHCUT_NUM_ITERS_PER_GLOBAL_RELABEL; i++) {
							gpuGraphCutPushKernel<<<gridSize, blockSize>>>(iters & 1, dGrid, dHeight, dExcess, dResidue);
							cudaThreadSynchronize();
							
							gpuGraphCutRelabelKernel<<<gridSize, blockSize>>>(iters & 1, dGrid, dHeight, dExcess, dResidue);
							cudaThreadSynchronize();
							iters++;
							
						}
						
						// Copy graph data back to CPU for global relabeling
						gpu.copyRawMat(&height, dHeight);
						gpu.copyRawMat(&excess, dExcess);
						gpu.copyRawMat(&residue, dResidue);
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

	template void gpuDecodeAlphaBeta<int>(Grid<int>*, Matrix<int>*, int, Matrix<int>*);
}
