#ifndef GRIDINF_INCLUDE_ESTIMATORS_H
#define GRIDINF_INCLUDE_ESTIMATORS_H

#include "core.h"

namespace ginf {
	// Find the labeling where every node chooses the label that minimizes the data cost
	template <typename T>
	void decodeMaxPrior(Grid<T> *grid, Matrix<int> *result) {
		for (int y = 0; y < grid->getHeight(); y++) {
			for (int x = 0; x < grid->getWidth(); x++) {
				int minLabel = 0;
				
				for (int i = 1; i < grid->getNumLabels(); i++) {
					if (grid->getDataCost(x, y, i) < grid->getDataCost(x, y, minLabel)) {
						minLabel = i;
					}
				}
				
				result->get(x, y) = minLabel;
			}
		}
	}
}

#endif
