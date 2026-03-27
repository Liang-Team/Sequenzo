/*
 * fastcluster_linkage_tu.cpp
 *
 * Separate translation unit compiled WITH -ffast-math.
 * Contains the fastcluster linkage bridge code:
 *   - fastcluster_linkage.cpp (which #includes fastcluster.cpp)
 *
 * The core linkage algorithms (NN_chain_core, MST_linkage_core, etc.)
 * benefit from -ffast-math optimizations (e.g., reciprocal approximation
 * for division in f_ward, auto-vectorization of distance update loops).
 *
 * NaN checking within the algorithms is controlled by skip_nan_check
 * parameter, and Sequenzo's pre-processing ensures no NaN values reach
 * these functions, so -ffast-math's NaN assumption is safe here.
 */
#include "fastcluster_linkage.cpp"
