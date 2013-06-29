#ifndef GRIDINF_INCLUDE_UTILS_H
#define GRIDINF_INCLUDE_UTILS_H

#define GINF_UNDEFINED -1

// Number of directions
#define GINF_NUM_DIR 4

// Returns the direction that is opposite to some specified direction
#define GINF_OPP_DIR(x) ((x) >= 2 ? (x) - 2 : (x) + 2)

// Returns true if a coordinate is within a rectangular bound
#define GINF_IS_VALID_NODE(x, y, w, h) (x >= 0 && y >= 0 && x < w && y < h)

// Declare useful arithmetic macros that we can use in both host and device code
#define GINF_MIN(a, b) ((a < b) ? (a) : (b))
#define GINF_MAX(a, b) ((a > b) ? (a) : (b))
#define GINF_ABS(x) ((x) < 0 ? (-(x)) : (x))
#define GINF_SQ(x) ((x) * (x))

// The maximum number of labels in a grid
#define GINF_MAX_NUM_LABELS 16

namespace ginf {
	// Enum of all the directions
	enum Directions {
		GINF_DIR_S,
		GINF_DIR_E,
		GINF_DIR_N,
		GINF_DIR_W
	};

	// Array of coordinate offsets in each direction
	// e.g. SOUTH is (0, 1), that is, the coordinate south of (x, y) is
	// (x, y + 1).s
	const int dirX[] = { 0, 1, 0, -1 };
	const int dirY[] = { 1, 0, -1, 0 };
}

#endif
