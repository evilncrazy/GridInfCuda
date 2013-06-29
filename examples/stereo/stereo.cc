#include "core.h"
#include "bp.h"

#include <cv.h>
#include <highgui.h>

#include <cmath>
#include <cstdio>

#define NUM_LABELS 16

using namespace cv;

int main(int argc, char *argv[]) {
	// Load in image using OpenCV
	Mat left = imread(argc == 3 ? argv[1] : "left.png", CV_LOAD_IMAGE_GRAYSCALE),
	    right = imread(argc == 3 ? argv[2] : "right.png", CV_LOAD_IMAGE_GRAYSCALE);
	
	if (!left.data || !right.data) {
		printf("Error in loading stereo image pair\n");
		return -1;
	}

	// Create a grid MRF for this image
	ginf::Grid<int> grid(left.cols, left.rows, NUM_LABELS);
	
	// Calculate the data costs for each pixel
	for (int y = 0; y < grid.getHeight(); y++) {
		for (int x = grid.getNumLabels() - 1; x < grid.getWidth(); x++) {
			for (int k = 0; k < NUM_LABELS; k++) {
				grid.setDataCost(x, y, k, std::min(abs(left.at<uchar>(y, x) - right.at<uchar>(y, x - k)), 20));
			}
		}
	}

	// Use a truncated linear model for the smoothness function
	grid.useSmoothnessTruncLinear(10, 20);

	// Create a 2D matrix to hold the final labels of each node
	ginf::Matrix<int> result(2, grid.getWidth(), grid.getHeight());

	// Decode using belief propagation for 30 iterations
	ginf::gpuDecodeBp(&grid, 30, &result);

	// Convert back into a OpenCV matrix
	Mat_<uchar> disparityMat(grid.getHeight(), grid.getWidth());
	for (int y = 0; y < grid.getHeight(); y++) {
		for (int x = 0; x < grid.getWidth(); x++) {
			disparityMat(y, x) = result.get(x, y) * (256 / NUM_LABELS);
		}
	}

	// Display the disparity result
	namedWindow("Disparity Result", CV_WINDOW_AUTOSIZE);
	imshow("Disparity Result", disparityMat);

	waitKey(0);
	return 0;
}
