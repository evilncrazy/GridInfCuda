#include "../include/bp.h"

namespace ginf {
	template <typename T>
	__device__ void gpuBpCollect(GpuGrid<T> *grid, int w, int h, T *msgs, T *from, int x, int y) {
		MatDim dim(w, h);
		MatDim dimMsgs(w, h, GINF_NUM_DIR);

		// For each state
		for (int i = 0; i < grid->getNumLabels(); i++) {
			from[dim.idx(x, y, i)] = 0;
		
			// For each neighbour in each direction, sum up the messages
			for (int d = 0; d < GINF_NUM_DIR; d++) {
				from[dim.idx(x, y, i)] += msgs[dimMsgs.idx(x + dDirX[d], y + dDirY[d], GINF_OPP_DIR(d), i)];
			}
		}
	}
	
	template <typename T>
	__device__ void gpuBpSendMsg(int iter, GpuGrid<T> *grid, int w, int h, T *msgs, T *from, T *data, T *out, int d) {
		GINF_DECL_ALT_X(iter); GINF_DECL_Y;
		MatDim dim(w, h);
		MatDim dimMsgs(w, h, GINF_NUM_DIR);

	   	int nx = x + dDirX[d], ny = y + dDirY[d];

		T minh = out[0] = data[dim.idx(x, y, 0)] + (from[dim.idx(x, y, 0)] - msgs[dimMsgs.idx(nx, ny, GINF_OPP_DIR(d), 0)]);
	    	for (int i = 1; i < grid->getNumLabels(); i++) {
	    		out[i] = data[dim.idx(x, y, i)] + (from[dim.idx(x, y, i)] - msgs[dimMsgs.idx(nx, ny, GINF_OPP_DIR(d), i)]);
	    		minh = GINF_MIN(minh, out[i]);
	    	}
	    	
	    	// Find the linear scaling constant
		T scale = grid->getSmoothnessCost(0, 1);
		for (int i = 1; i < grid->getNumLabels(); i++)
			out[i] = GINF_MIN(out[i], out[i - 1] + scale);
		for (int i = grid->getNumLabels() - 2; i >= 0; i--)
			out[i] = GINF_MIN(out[i], out[i + 1] + scale);
	    	
		// Truncate the messages using truncate constant
		minh += GINF_MIN(grid->getSmoothnessCost(0, 1) * grid->getNumLabels(),
				grid->getSmoothnessCost(0, grid->getNumLabels() - 1));
		T val = 0;
		for (int i = 0; i < grid->getNumLabels(); i++) {
			out[i] = GINF_MIN(out[i], minh);
			val += out[i];
		}
		
		// Normalize
		val /= grid->getNumLabels();
		for (int i = 0; i < grid->getNumLabels(); i++) {
			out[i] -= val;
		}
	}
	
	template <typename T>
	__global__ void gpuBpKernel(int iter, GpuGrid<T> *grid, int w, int h, T *msgs, T *from, T *data) {
		GINF_DECL_ALT_X(iter); GINF_DECL_Y;
		MatDim dim(w, h, GINF_NUM_DIR);
		
		if (dim.isValidStrict(x, y)) {
			// Collect messages from neighbours
			gpuBpCollect(grid, w, h, msgs, from, x, y);
			
			// Foreach direction, send message
			T out[GINF_MAX_NUM_LABELS];
			for (int d = 0; d < GINF_NUM_DIR; d++) {
				gpuBpSendMsg(iter, grid, w, h, msgs, from, data, out, d);
				
				for (int i = 0; i < grid->getNumLabels(); i++) {
					msgs[dim.idx(x, y, d, i)] = out[i];
				}
			}
		}
	}

	template <typename T>
	__global__ void gpuBpGetBeliefKernel(GpuGrid<T> *grid, T *msgs, T *from, int *result) {
		GINF_DECL_X; GINF_DECL_Y;
		MatDim dim(grid->getWidth(), grid->getHeight());
		MatDim dimMsgs(grid->getWidth(), grid->getHeight(), GINF_NUM_DIR);
		
		if (dim.isValidStrict(x, y)) {
			// Collect msgs from neighbours
			gpuBpCollect(grid, grid->getWidth(), grid->getHeight(), msgs, from, x, y);
		
			// Pick the label that minimizes the cost
			T minCost = grid->getDataCost(x, y, 0) + from[dim.idx(x, y, 0)];
			int minLabel = 0;
			
			for (int i = 1; i < grid->getNumLabels(); i++) {
				T cost = grid->getDataCost(x, y, i) + from[dim.idx(x, y, i)];
				if (cost < minCost) {
					minCost = cost;
					minLabel = i;
				}
			}
			
			// Write the result
			result[dim.idx(x, y)] = minLabel;
		}
	}
	
	/*
	template <typename T>
	__global__ void gpuBpCalcDataCost(GpuGrid<T> *grid, int w, int h, int pw, int ph, T *data, T *prev) {
		GINF_DECL_X; GINF_DECL_Y;
		MatDim dim(w, h, grid->getNumLabels());
		MatDim dimPrev(pw, ph, grid->getNumLabels());
		
		if (dim.isValid(x / 2, y / 2)) {
			for (int i = 0; i < grid->getNumLabels(); i++) {
				data[dim.idx(x / 2, y / 2, i)] += prev[dimPrev.idx(x, y, i)];
			}
		}
	}*/
	
	template <typename T>
	__global__ void gpuBpInitMsgs(GpuGrid<T> *grid, int w, int h, int pw, int ph, T *msgs, T *prev) {
		GINF_DECL_X; GINF_DECL_Y;	
		MatDim dim(w, h, GINF_NUM_DIR);
		MatDim dimPrev(pw, ph, GINF_NUM_DIR);
		
		if (dim.isValidStrict(x, y)) {
			for (int d = 0; d < GINF_NUM_DIR; d++) {
				for (int i = 0; i < grid->getNumLabels(); i++) {
					// Initialize message with node on the above level
					msgs[dim.idx(x, y, d, i)] = prev[dimPrev.idx(x / 2, y / 2, d, i)];
				}
			}
		}
	}

	template <typename T>
	void gpuDecodeHbp(Grid<T> *grid, int numLevels, int numItersPerLevel, Matrix<int> *result) {
		Gpu gpu;
		
		// Create matrices to store data costs at each level
            	Matrix<T> **data = new Matrix<T>* [numLevels];
		
		// Allocate memory on the device
		int *dResult = gpu.createRawMat(result);
		
		// Each of these is an array of device pointers to matrices, one for each level.
		T **dMsgs = new T*[numLevels],
		  **dFrom = new T*[numLevels],
		  **dData = new T*[numLevels];
		
		GpuGrid<T> *dGrid = gpu.createGrid(grid);
		
		// At the finest level, the data costs correspond to the original problem
		data[0] = new Matrix<T>(3, grid->getWidth(), grid->getHeight(), grid->getNumLabels());
		data[0]->copyFrom(grid->getDataCosts());
		
		// Copy it to the device
		dData[0] = gpu.createRawMat(data[0]);
           
           	// Prepare block sizes
		dim3 blockSize(GINF_BP_BLOCK_SIZE, GINF_BP_BLOCK_SIZE);
		dim3 gridSize(GINF_DIM(grid->getWidth(), GINF_BP_BLOCK_SIZE), 
			      GINF_DIM(grid->getHeight(), GINF_BP_BLOCK_SIZE));

		// Calculate the data costs on the other levels
		for (int t = 1; t < numLevels; t++) {
			// Calculate the dimensions of the grid at this level
			int w = (int)ceil(data[t - 1]->getSize(0) / 2.0), h = (int)ceil(data[t - 1]->getSize(1) / 2.0);

			// We gotta make sure that we don't get too 'coarse'
			if (w < 1 || h < 1) return;

			// The data cost of each node on the current level uses the sum of
			// the data costs of four nodes below this level
			data[t] = new Matrix<T>(3, w, h, grid->getNumLabels());
			for (int y = 0; y < data[t - 1]->getSize(1); y++) {
				for (int x = 0; x < data[t - 1]->getSize(0); x++) {
					for (int i = 0; i < grid->getNumLabels(); i++) {
						data[t]->get(x / 2, y / 2, i) += data[t - 1]->get(x, y, i);
					}
				}
			}
			
			// Copy data cost matrix on the device
			dData[t] = gpu.createRawMat(data[t]);
			//gpuBpCalcDataCost<<<gridSize, blockSize>>>(dGrid, w, h, data[t - 1]->getSize(0), data[t - 1]->getSize(1), dData[t], dData[t - 1]);
		}
		
		// For each level, we'll do message passing
		for (int t = numLevels - 1; t >= 0; t--) {
			int w = data[t]->getSize(0), h = data[t]->getSize(1);
			
			// Allocate message matrices on the device for this particular level
			dMsgs[t] = gpu.createRawMat<T>(w * h * GINF_NUM_DIR * grid->getNumLabels());
			dFrom[t] = gpu.createRawMat<T>(w * h * grid->getNumLabels());

			// If we're not on the top level, the messages are initialized from the previous level
			if (t != numLevels - 1) {
				int pw = data[t + 1]->getSize(0), ph = data[t + 1]->getSize(1);
				
				gpuBpInitMsgs<<<gridSize, blockSize>>>(dGrid, w, h, pw, ph, dMsgs[t], dMsgs[t + 1]);

				// Free up memory
				delete data[t + 1];
			}
			
			// For each iteration, run belief propagation
			for (int i = 0; i < numItersPerLevel; i++) {
				gpuBpKernel<<<gridSize, blockSize>>>(i, dGrid, w, h, dMsgs[t], dFrom[t], dData[t]);
			}
		}
		
		// Finally get the beliefs
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
	void gpuDecodeBpSync(Grid<T> *grid, int numIters, Matrix<int> *result) {
		gpuDecodeHbp(grid, 1, numIters, result);
	}
	
	template void gpuDecodeBpSync<float>(Grid<float>*, int, Matrix<int>*);
	template void gpuDecodeBpSync<int>(Grid<int>*, int, Matrix<int>*);
}
