/*
 * fastcluster_tu.cpp
 *
 * Separate translation unit compiled WITHOUT -ffast-math.
 * Contains code that requires IEEE NaN/Inf semantics:
 *   - fastcluster (fc_isnan depends on NaN != NaN)
 *   - distance_prep_utils (NaN detection/replacement)
 */
#include "distance_prep_utils.cpp"
#include "fastcluster_linkage.cpp"
