#include "include/bp.h"

#include <cmath>

namespace ginf {
	// Sum up messages sent to a node by its neighbours
	template <typename T>
	void bpCollect(Grid<T> *grid, Matrix<T> *msgs, Matrix<T> *from, Matrix<T> *dataCosts, int x, int y) {
		// For each label
		for (int i = 0; i < grid->getNumLabels(); i++) {
			from->get(x, y, i) = dataCosts->at(x, y, i);
			
			// For each neighbour, add up the messages for this particular label
			for (int d = 0; d < GINF_NUM_DIR; d++) {
				from->get(x, y, i) += msgs->at(x + dirX[d], y + dirY[d], GINF_OPP_DIR(d), i);
			}
		}
	}

	// Send a message in a particular direction
	template <typename T>
	void bpSendMsg(Grid<T> *grid, Matrix<T> *msgs, Matrix<T> *from, T *out, int x, int y, int d) {
		// Get the coordinate of the neighbour
		int nx = x + dirX[d], ny = y + dirY[d];

		if (grid->getSmModel() == GINF_SM_TRUNC_LINEAR || grid->getSmModel() == GINF_SM_POTTS) {
			// We can apply an optimization to calculate the message in O(L), where L is the number of labels
			T minh = out[0] = from->at(x, y, 0) - msgs->at(nx, ny, GINF_OPP_DIR(d), 0);
			for (int i = 1; i < grid->getNumLabels(); i++) {
				out[i] = from->at(x, y, i) - msgs->at(nx, ny, GINF_OPP_DIR(d), i);
				minh = GINF_MIN(minh, out[i]);
			}

			// Find the linear scaling constant
			T scale = grid->getSmoothnessCost(0, 1);
			for (int i = 1; i < grid->getNumLabels(); i++)
				out[i] = GINF_MIN(out[i], out[i - 1] + scale);

			for (int i = grid->getNumLabels() - 2; i >= 0; i--)
				out[i] = GINF_MIN(out[i], out[i + 1] + scale);

			// Truncate the messages using truncate constant
			minh += grid->getSmoothnessCost(0, grid->getNumLabels() - 1);
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
		} else {
			for (int i = 0; i < grid->getNumLabels(); i++) {
				out[i] = grid->getSmoothnessCost(i, 0) + from->at(x, y, 0);
				for (int j = 0; j < grid->getNumLabels(); j++) {
					out[i] = GINF_MIN(out[i], grid->getSmoothnessCost(i, j) + from->at(x, y, j));
				}
			}
		}
	}

	template <typename T>   
	int bpGetBelief(Grid<T> *grid, Matrix<T> *msgs, Matrix<T> *from, Matrix<T> *data, int x, int y) {
		// Collect msgs from neighbours
		bpCollect(grid, msgs, data, from, x, y);
	
		// Pick the label that minimizes the cost
		T minCost = from->at(x, y, 0);
		int minLabel = 0;
		
		for (int i = 1; i < grid->getNumLabels(); i++) {
			T cost = from->at(x, y, i);
			if (cost < minCost) {
				minCost = cost;
				minLabel = i;
			}
		}
		
		// Return the result
		return minLabel;
	}

	template <typename T>
	void decodeHbp(Grid<T> *grid, int numLevels, int numItersPerLevel, Matrix<int> *result) {	
		// Create matrices to store the messages for each level
		Matrix<T> **msgs = new Matrix<T>* [numLevels];
		Matrix<T> **from = new Matrix<T>* [numLevels];

		// Create matrices to store data costs at each level
		Matrix<T> **data = new Matrix<T>* [numLevels];

		// At the finest level, the data costs correspond to the original problem
		data[0] = new Matrix<T>(3, grid->getWidth(), grid->getHeight(), grid->getNumLabels());
		data[0]->copyFrom(grid->getDataCosts());
	   	
		// Calculate the data costs on the other levels
		for (int t = 1; t < numLevels; t++) {
			int w = (int)ceil(data[t - 1]->getSize(0) / 2.0),
			h = (int)ceil(data[t - 1]->getSize(1) / 2.0);

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
		}

		// Use a temporary array to store message that is sent from one particular node to another
		T *msgBuffer = new T[grid->getNumLabels()];
		for (int t = numLevels - 1; t >= 0; t--) {
			int w = data[t]->getSize(0), h = data[t]->getSize(1);

			// Create the msg matrices for this particular level
			msgs[t] = new Matrix<T>(4, w, h, GINF_NUM_DIR, grid->getNumLabels());
			from[t] = new Matrix<T>(3, w, h, grid->getNumLabels());
	   
			// If we're not on the top level, the messages are initialized from the previous level
			if (t != numLevels - 1) {
				for (int y = 0; y < h; y++) {
					for (int x = 0; x < w; x++) {
						for (int d = 0; d < GINF_NUM_DIR; d++) {
							for (int i = 0; i < grid->getNumLabels(); i++) {
								// Initialize message with node on the above level
								msgs[t]->get(x, y, d, i) = msgs[t + 1]->get(x / 2, y / 2, d, i);
							}
						}
					}
				}
				
				// Free up memory
				delete msgs[t + 1];
				delete data[t + 1];
				delete from[t + 1];
			}

			// For each level, perform belief propagation using checkerboard pattern
			for (int i = 0; i < numItersPerLevel; i++) {
				for (int y = 1; y < h - 1; y++) {
					for (int x = ((y + i) & 1) + 1; x < w - 1; x += 2) {
						// Collect msgs from neighbours
						bpCollect(grid, msgs[t], from[t], data[t], x, y);

						// Foreach direction, send message
						for (int d = 0; d < GINF_NUM_DIR; d++) {
							bpSendMsg(grid, msgs[t], from[t], msgBuffer, x, y, d);
	
							// Copy the msgBuffer into the msgs array
							for (int j = 0; j < grid->getNumLabels(); j++) {
								msgs[t]->get(x, y, d, j) = msgBuffer[j];
							}
						}
					}
				}
			}
		}

		// We won't need this anymore
		delete[] msgBuffer;

		// Finally, get the belief of every node
		for (int y = 1; y < grid->getHeight() - 1; y++) {
			for (int x = 1; x < grid->getWidth() - 1; x++) {
				result->get(x, y) = bpGetBelief(grid, msgs[0], from[0], data[0], x, y);
			}
		}
	}
	
	template <typename T>
	void decodeBpSync(Grid<T> *grid, int numIters, Matrix<int> *result) {
		// Synchronous loopy belief propagation is equivalent to running
		// hierarchical belief propagation for one level
		decodeHbp(grid, 1, numIters, result);
	}
	
	// Instantiate templates
	template void decodeHbp<float>(Grid<float>*, int, int, Matrix<int>*);
	template void decodeHbp<int>(Grid<int>*, int, int, Matrix<int>*);

	template void decodeBpSync<float>(Grid<float>*, int, Matrix<int>*);
	template void decodeBpSync<int>(Grid<int>*, int, Matrix<int>*);
}

