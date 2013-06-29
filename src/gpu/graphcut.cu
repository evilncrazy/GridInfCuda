#include "../include/graphcut.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace ginf {
	template <typename T>
	__device__ void gpuGraphCutPushToPixel(GpuGrid<T> *grid, T *excess, T *residue, int d) {
		GINF_DECL_X; GINF_DECL_Y;
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
	__global__ void gpuGraphCutPushKernel(GpuGrid<T> *grid, int *height, T *excess, T *residue) {
		GINF_DECL_X; GINF_DECL_Y;
		MatDim dim(grid->getWidth(), grid->getHeight());

		if (dim.isValid(x, y) && excess[dim.idx(x, y)] && height[dim.idx(x, y)] < grid->getNumNodes() + 2) {
			// Try to push to the sink
			if (residue[dim.idx(x, y, GINF_DIR_SINK)] && height[dim.idx(x, y)] == 1) {
				T flow = GINF_MIN(residue[dim.idx(x, y, GINF_DIR_SINK)], excess[dim.idx(x, y)]);

				excess[dim.idx(x, y)] -= flow;
				residue[dim.idx(x, y, GINF_DIR_SINK)] -= flow;
			}
		
			// Push to neighbours
			for (int d = 0; d < GINF_NUM_DIR; d++) {
				if (residue[dim.idx(x, y, d)] && height[dim.idx(x, y)] == height[dim.idx(x + dDirX[d], y + dDirY[d])] + 1) {
					gpuGraphCutPushToPixel(grid, excess, residue, d);
				}
			}
		}
	}
	
	template <typename T>
	__global__ void gpuGraphCutRelabelKernel(GpuGrid<T> *grid, int *height, T *excess, T *residue) {
		GINF_DECL_X; GINF_DECL_Y;
		MatDim dim(grid->getWidth(), grid->getHeight());

		if (dim.isValid(x, y) && excess[dim.idx(x, y)] && height[dim.idx(x, y)] < grid->getNumNodes() + 2) {
			if (residue[dim.idx(x, y, GINF_DIR_SINK)]) {
				height[dim.idx(x, y)] = 1;
			} else {
				int minHeight = grid->getNumNodes() + 2;
				for (int d = 0; d < GINF_NUM_DIR; d++) {
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
		if (dim.isValid(x, y) && (f[dim(x, y)] == alpha || f[dim(x, y)] == beta)) {
			int alphaCost = 0, betaCost = 0;
			for (int d = 0; d < GINF_NUM_DIR; d++) {
				int nx = x + dDirX[d], ny = y + dDirY[d];
				if (dim.isValid(nx, ny)) {
					// Check if this neighbour (nx, ny) is in the alpha-beta partition
					if (f[dim(nx, ny)] == alpha || f[dim(nx, ny)] == beta) {
						// Capacity of this edge is V(alpha, beta)
						capacity[dim.idx(x, y, d)] = grid->getSmoothnessCost(alpha, beta);
					} else {
						// There is no edge between these two nodes
						capacity[dim.idx(x, y, d)] = 0;

						alphaCost += grid->getSmoothnessCost(alpha, f[dim(nx, ny)]);
						betaCost += grid->getSmoothnessCost(beta, f[dim(nx, ny)]);
					}
				} else {
					// No edge here
					capacity[dim(x, y, d)] = 0;
				}
			}

			// Calculate the capacity of t-links
			excess[dim(x, y)] = grid->getDataCost(x, y, alpha) + alphaCost;
			capacity[dim(x, y, GINF_DIR_SINK)] = grid->getDataCost(x, y, beta) + betaCost;

			// Reset height to 0
			height[dim(x, y)] = 0;
		}
	}
	
	template <typename T>
	__global__ void gpuFindActiveTiles(GpuGrid<T> *grid, int *height, T *excess, int *active) {
		GINF_DECL_X; GINF_DECL_Y;
		MatDim dim(grid->getWidth(), grid->getHeight());
		
		if (dim.isValid(x, y)) {
			if (excess[dim(x, y)] && height[dim(x, y)] < grid->getNumNodes() + 2)
				active[blockIdx.x + blockIdx.y * gridDim.x] = 1;
		}
	}
	
	int buildActiveTileList(Matrix<int> *active, int2 *list) {
		int count = 0;
		
		for (int y = 0; y < active->getSize(1); y++) {
			for (int x = 0; x < active->getSize(0); x++) {
				if (active->at(x, y)) {
					list[count++] = make_int2(x, y);
				}
			}
		}
		
		return count;
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
		
		// Prepare block sizes
		dim3 blockSize(GINF_GRAPHCUT_BLOCK_WIDTH, GINF_GRAPHCUT_BLOCK_HEIGHT);
		dim3 gridSize(GINF_DIM(grid->getWidth(), blockSize.x), GINF_DIM(grid->getHeight(), blockSize.y));
		
		// Boolean matrix indicating whether a node is reachable from the source in the residue graph
		Matrix<int> reachable(2, grid->getWidth(), grid->getHeight());
		
		// Boolean matrix indicate whether a block contains an active node or not
		Matrix<int> isActive(2, gridSize.x, gridSize.y);
		
		// List of active blocks (xy coordinates)
		int2 activeList[gridSize.x * gridSize.y];

		// Create a temp matrix to store new labelings
		Matrix<int> newResult(2, grid->getWidth(), grid->getHeight());
		newResult.copyFrom(initial);

		// Allocate GPU memory and transfer data
		T *dExcess = gpu.createRawMat(&excess),
		  *dResidue = gpu.createRawMat(&residue);
		int *dResult = gpu.createRawMat(result),
		  *dHeight = gpu.createRawMat(&height),
		  *dActive = gpu.createRawMat(&isActive);
		
		GpuGrid<T> *dGrid = gpu.createGrid(grid);
		
		// Loop until there are no more label changes
		bool success = true;
		for (int t = 0; t < numIters && success; t++) {
			success = false;

			// Loop through each pair of labels
			T minCost = grid->getLabelingCost(result);
			for (int alpha = 0; alpha < grid->getNumLabels(); alpha++) {
				printf("%lf,%d\n", TIMER_END, grid->getLabelingCost(result));
			
				for (int beta = alpha + 1; beta < grid->getNumLabels(); beta++) {
					// Now, construct the graph with alpha and beta as terminal nodes
					gpuGraphCutConstructGraphKernel<<<gridSize, blockSize>>>(dGrid, dResult, dHeight, dExcess, dResidue, alpha, beta);

					// We need to keep this data synced between CPU and GPU
					// every time we perform global relabeling
					gpu.copyRawMat(&height, dHeight);
					gpu.copyRawMat(&residue, dResidue);

					// Run push-relabel until convergence
					int iters = 0;
					while (true) {
						// Mark each tile as active or inactive
						gpu.clearMat(dActive, isActive.getTotalSize());
						gpuFindActiveTiles<<<gridSize, blockSize>>>(dGrid, dHeight, dExcess, dActive);
						gpu.copyRawMat(&isActive, dActive);
						
						// Build a list of active tiles
						if (buildActiveTileList(&isActive, activeList) == 0) break;
						
						// Global relabeling optimization on the CPU
						graphCutGlobalRelabel(grid, &height, &residue);
						
						// Copy relabeled height data back to the GPU
						gpu.copyRawMat(dHeight, &height);

						// Run normal push and relabel on the GPU
						for (int i = 0; i < GINF_GRAPHCUT_NUM_ITERS_PER_GLOBAL_RELABEL; i++) {
							gpuGraphCutPushKernel<<<gridSize, blockSize>>>(dGrid, dHeight, dExcess, dResidue);
							gpuGraphCutRelabelKernel<<<gridSize, blockSize>>>(dGrid, dHeight, dExcess, dResidue);
							
							iters++;
						}
						
						// Copy graph data back to CPU for global relabeling
						gpu.copyRawMat(&height, dHeight);
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
						
						// Copy result to GPU as well
						gpu.copyRawMat(dResult, result);
						
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
