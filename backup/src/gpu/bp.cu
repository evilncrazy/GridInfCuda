#include "../include/bp.h"

namespace ginf {
	template <typename T>
	__global__ void gpuBpCollectKernel(GpuGrid<T> *grid, int w, int h, T *msgs, T *from, T *data) {
		GINF_DECL_X; GINF_DECL_Y;
	
		MatDim dim(w, h);
		MatDim dimMsgs(w, h, GINF_NUM_DIR);

		if (dim.isValidStrict(x, y)) {
			// For each state
			for (int i = 0; i < grid->getNumLabels(); ++i) {
				from[dim(x, y, i)] = data[dim(x, y, i)];
				
				// For each neighbour in each direction, sum up the messages
				for (int d = 0; d < GINF_NUM_DIR; d++) {
					from[dim(x, y, i)] += msgs[dimMsgs(x, y, d, i)];
				}
			}
		}
	}
	
	template <typename T>
	__global__ void gpuBpKernel(int iter, GpuGrid<T> *grid, int w, int h, T *msgs, T *from, T *data, int d) {
		GINF_DECL_ALT_X(iter); GINF_DECL_Y;
		
		MatDim dim(w, h, GINF_NUM_DIR);
		MatDim dimMsgs(w, h, GINF_NUM_DIR);

		if (dim.isValidStrict(x, y)) {
			// Create a vector to store the message. This is stored in local memory, and so writing to it
			// is coalesced.
			T out[GINF_MAX_NUM_LABELS];
			
			// Find the linear scaling constant
			T scale = grid->getSmoothnessCost(0, 1);
			
			// Calculate message vector in the forward direction
			T minh = out[0] = from[dim(x, y, 0)] - msgs[dimMsgs(x, y, d, 0)];
			for (int i = 1; i < grid->getNumLabels(); i++) {
				out[i] = GINF_MIN(from[dim(x, y, i)] - msgs[dimMsgs(x, y, d, i)], out[i - 1] + scale);
				minh = GINF_MIN(minh, out[i]);
			}
				
			// Truncate the messages using truncate constant
			minh += grid->getSmoothnessCost(0, grid->getNumLabels() - 1);
				
			// Compute the final out vector, and sum up all the elements to perform normalization
			T val = 0;
			for (int i = grid->getNumLabels() - 2; i >= 0; i--) {
				out[i] = GINF_MIN(GINF_MIN(out[i], out[i + 1] + scale), minh);
				val += out[i];
			}

			// Find the normalizing constant
			val /= grid->getNumLabels();

			// Update the message
			int nx = x + dDirX[d], ny = y + dDirY[d];
			for (int i = 0; i < grid->getNumLabels(); i++) {
				// Update using the normalized message values
				msgs[dim(nx, ny, GINF_OPP_DIR(d), i)] = out[i] - val;
			}
		}
	}

	template <typename T>
	__global__ void gpuBpGetBeliefKernel(GpuGrid<T> *grid, T *msgs, T *from, int *result) {
		GINF_DECL_X; GINF_DECL_Y;
		MatDim dim(grid->getWidth(), grid->getHeight());
		
		if (dim.isValidStrict(x, y)) {
			// Pick the label that minimizes the cost
			T minCost = from[dim(x, y, 0)];
			int minLabel = 0;
			
			for (int i = 1; i < grid->getNumLabels(); ++i) {
				T cost = from[dim(x, y, i)];
				if (cost < minCost) {
					minCost = cost;
					minLabel = i;
				}
			}
			
			// Write the result
			result[dim(x, y)] = minLabel;
		}
	}
	
	template <typename T>
	__global__ void gpuBpCalcDataCost(GpuGrid<T> *grid, int w, int h, int pw, int ph, T *data, T *prev) {
		GINF_DECL_X; GINF_DECL_Y;
		MatDim dim(w, h, grid->getNumLabels());
		MatDim dimPrev(pw, ph, grid->getNumLabels());

		if (dim.isValid(x, y)) {
			for (int i = 0; i < grid->getNumLabels(); i++) {
				data[dim(x, y, i)] = prev[dimPrev(x * 2, y * 2, i)];
				data[dim(x, y, i)] += prev[dimPrev(x * 2 + 1, y * 2, i)];
				data[dim(x, y, i)] += prev[dimPrev(x * 2, y * 2 + 1, i)];
				data[dim(x, y, i)] += prev[dimPrev(x * 2 + 1, y * 2 + 1, i)];
			
				//data[dim.idx(x / 2, y / 2, i)] += prev[dimPrev.idx(x, y, i)];
			}
		}
	}
	
	template <typename T>
	__global__ void gpuBpInitMsgs(GpuGrid<T> *grid, int w, int h, int pw, int ph, T *msgs, T *prev) {
		GINF_DECL_X; GINF_DECL_Y;
		
		MatDim dim(w, h, GINF_NUM_DIR);
		MatDim dimPrev(pw, ph, GINF_NUM_DIR);
		
		// We will initialize 4 messages per thread, because they all will be initialized
		// to the same value anyway.
		x *= 2; y *= 2;
		
		if (dim.isValidStrict(x, y)) {
			for (int d = 0; d < GINF_NUM_DIR; d++) {
				for (int i = 0; i < grid->getNumLabels(); i++) {
					// Initialize message with node on the above level
					msgs[dim(x, y, d, i)] =
					msgs[dim(x + 1, y, d, i)] =
					msgs[dim(x, y + 1, d, i)] =
					msgs[dim(x + 1, y + 1, d, i)] = prev[dimPrev(x / 2, y / 2, d, i)];
				}
			}
		}
	}

	template <typename T>
	void gpuDecodeHbp(Grid<T> *grid, int numLevels, int numItersPerLevel, Matrix<int> *result) {
		Gpu gpu;
		
		// We need to keep track of the widths and heights of each level
		int *widths = new int[numLevels], *heights = new int[numLevels];
		
		// Allocate memory on the device
		int *dResult = gpu.createRawMat(result);
		
		// Each of these is an array of device pointers to matrices, one for each level.
		T **dMsgs = new T*[numLevels],
		  **dFrom = new T*[numLevels],
		  **dData = new T*[numLevels];
		
		GpuGrid<T> *dGrid = gpu.createGrid(grid);
		
		// At the finest level, the data costs correspond to the original problem
		widths[0] = grid->getWidth();
		heights[0] = grid->getHeight();

		// Copy the original data costs to the device
		dData[0] = gpu.createRawMat(grid->getDataCosts());

		// Prepare block sizes
		dim3 blockSize(GINF_BP_BLOCK_WIDTH, GINF_BP_BLOCK_HEIGHT);
		dim3 gridSize(GINF_DIM(grid->getWidth(), blockSize.x), GINF_DIM(grid->getHeight(), blockSize.y));

		// Calculate the data costs on the other levels
		for (int t = 1; t < numLevels; t++) {
			// Calculate the dimensions of the grid at this level
			int w = widths[t] = (int)ceil(widths[t - 1] / 2.0);
			int h = heights[t] = (int)ceil(heights[t - 1] / 2.0);

			// We gotta make sure that we don't get too 'coarse'
			if (w < 1 || h < 1) {
				numLevels = t - 1;
				break;
			}

			// The data cost of each node on the current level uses the sum of the 
			// data costs of four nodes below this level. We calculate this in a 
			dData[t] = gpu.createRawMat<T>(w * h * grid->getNumLabels());
			gpuBpCalcDataCost<<<gridSize, blockSize>>>(dGrid, w, h, widths[t - 1], heights[t - 1], dData[t], dData[t - 1]);
		}
		
		// For each level, we'll do message passing
		for (int t = numLevels - 1; t >= 0; t--) {
			int w = widths[t], h = heights[t];
			
			// Allocate message matrices on the device for this particular level
			dMsgs[t] = gpu.createRawMat<T>(w * h * GINF_NUM_DIR * grid->getNumLabels());
			dFrom[t] = gpu.createRawMat<T>(w * h * grid->getNumLabels());
			
			// Calculate gridSizes
			dim3 lvlGridSize(GINF_DIM(w, blockSize.x), GINF_DIM(h, blockSize.y));
			dim3 lvlGridSizeBipartite(GINF_DIM(w / 2 + 1, blockSize.x), GINF_DIM(h, blockSize.y));

			// If we're not on the top level, the messages are initialized from the previous level
			if (t != numLevels - 1) {
				gpuBpInitMsgs<<<lvlGridSize, blockSize>>>(dGrid, w, h, widths[t + 1], heights[t + 1], dMsgs[t], dMsgs[t + 1]);
			}
			
			// For each iteration, run belief propagation
			for (int i = 0; i < numItersPerLevel; i++) {
				gpuBpCollectKernel<<<lvlGridSize, blockSize>>>(dGrid, w, h, dMsgs[t], dFrom[t], dData[t]);

				// For each direction, send a message
				for (int d = 0; d < GINF_NUM_DIR; d++) {
					gpuBpKernel<<<lvlGridSizeBipartite, blockSize>>>(i, dGrid, w, h, dMsgs[t], dFrom[t], dData[t], d);
				}
			}
		}
		
		// Finally get the beliefs
		gpuBpCollectKernel<<<gridSize, blockSize>>>(dGrid, grid->getWidth(), grid->getHeight(), dMsgs[0], dFrom[0], dData[0]);
		gpuBpGetBeliefKernel<<<gridSize, blockSize>>>(dGrid, dMsgs[0], dFrom[0], dResult);
		
		// Transfer the results back to CPU
		gpu.copyRawMat(result, dResult);
		
		// Free GPU memory
		for (int i = 0; i < numLevels; i++) {
			gpu.free(dMsgs[i]);
			gpu.free(dFrom[i]);
			gpu.free(dData[i]);
		}
		
		gpu.free(dResult);
		gpu.free(dGrid);
	}
	
	template void gpuDecodeHbp<float>(Grid<float>*, int, int, Matrix<int>*);
	template void gpuDecodeHbp<int>(Grid<int>*, int, int, Matrix<int>*);
	
	template <typename T>
	void gpuDecodeBp(Grid<T> *grid, int numIters, Matrix<int> *result) {
		gpuDecodeHbp(grid, 1, numIters, result);
	}
	
	template void gpuDecodeBp<float>(Grid<float>*, int, Matrix<int>*);
	template void gpuDecodeBp<int>(Grid<int>*, int, Matrix<int>*);
}
