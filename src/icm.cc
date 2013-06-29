#include "include/icm.h"

namespace ginf {
	template <typename T>
	void decodeIcmSync(Grid<T> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result) {
		// Start with initial configuration
		result->copyFrom(initial);

		// Create another 2D matrix to store the labeling (synchronous updates)
		Matrix<int> oldLabels(2, grid->getWidth(), grid->getHeight());
		oldLabels.copyFrom(result);

		// For each iteration, set each node's label to one that minimizes the
		// sum of its data cost and its smoothness costs
		for (int t = 0; t < numIters; t++) {
			for (int y = 1; y < grid->getHeight() - 1; y++) {
				for (int x = 1; x < grid->getWidth() - 1; x++) {
					// Find the minimum state
					T minCost = grid->getLabelingCost(&oldLabels, x, y, 0);
					int minLabel = 0;
					
					for (int i = 1; i < grid->getNumLabels(); i++) {
						T cost = grid->getLabelingCost(&oldLabels, x, y, i);
						if (cost < minCost) {
							minLabel = i;
							minCost = cost;
						}
					}
					
					// Update this node's label
					result->get(x, y) = minLabel;
				}
			}

			// Copy labeling over to old labeling
			oldLabels.copyFrom(result);
		}
	}
	
	template <typename T>
	void decodeIcmAsync(Grid<T> *grid, Matrix<int> *initial, int numIters, Matrix<int> *result) {
		// Start with initial configuration
		result->copyFrom(initial);

		// For each iteration, set each node's label to one that minimizes the
		// sum of its data cost and its smoothness costs
		for (int t = 0; t < numIters; t++) {
			for (int y = 1; y < grid->getHeight() - 1; y++) {
				for (int x = 1; x < grid->getWidth() - 1; x++) {
					// Find the minimum state
					T minCost = grid->getLabelingCost(result, x, y, 0);
					int minLabel = 0;
					
					for (int i = 1; i < grid->getNumLabels(); i++) {
						T cost = grid->getLabelingCost(result, x, y, i);
						if (cost < minCost) {
							minLabel = i;
							minCost = cost;
						}
					}
					
					// Update this node's label
					result->get(x, y) = minLabel;
				}
			}
		}
	}
	
	// Instantiate templates
	template void decodeIcmSync<float>(Grid<float>*, Matrix<int>*, int, Matrix<int>*);
	template void decodeIcmSync<int>(Grid<int>*, Matrix<int>*, int, Matrix<int>*);

	template void decodeIcmAsync<float>(Grid<float>*, Matrix<int>*, int, Matrix<int>*);
	template void decodeIcmAsync<int>(Grid<int>*, Matrix<int>*, int, Matrix<int>*);
}
