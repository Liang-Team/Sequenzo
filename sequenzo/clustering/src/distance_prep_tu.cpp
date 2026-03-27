/*
 * distance_prep_tu.cpp
 *
 * Separate translation unit compiled WITHOUT -ffast-math.
 * Contains code that requires IEEE NaN/Inf semantics:
 *   - distance_prep_utils (NaN detection/replacement, std::isfinite)
 */
#include "distance_prep_utils.cpp"
